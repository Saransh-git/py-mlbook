import math, random


################################################################################
# Part 0: Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']


def start_pad(n):
    """ Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams """
    return '~' * n


def ngrams(n, text):
    """ Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character """
    n_grams = []
    for index, c in enumerate(text):
        n_grams.append((f"{start_pad(n-index)}{text[max(0, index-n):index]}", c))  # pads only the characters remaining
        # from the index considered and rest from text before that index
    return n_grams


def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model


def create_ngram_model_lines(model_class, path, n=2, k=0):
    """ Creates and returns a new n-gram model trained on the city names
        found in the path file """
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip())
    return model


################################################################################
# Part 1: Basic N-Gram Model
################################################################################

class NgramModel(object):
    """ A basic n-gram model using add-k smoothing """

    def __init__(self, n, k):
        self.n = n
        self.vocab_set = set()  # maintains the vocab set
        self.is_vocab_sorted = True  # to denote if the vocab is sorted or not
        self.context_dict = {}  # for maintaining counts, is a dict of dict wherein, the key is context and the value is
        # a dict of char and its corresponding count
        self.k = k  # stores k for add-k smoothing

    def get_vocab(self):
        """ Returns the set of characters in the vocab """
        return self.vocab_set

    def update(self, text):
        """ Updates the model n-grams based on text """
        self.vocab_set.update(set(text))
        self.is_vocab_sorted = False  # make this False to denote vocab is not sorted
        self.populate_context_dict(self.n, text)

    def populate_context_dict(self, n, text):
        """
        Populates context dict with the n-gram contexts and char counts
        """
        n_grams = ngrams(n, text)
        for gram in n_grams:
            context = gram[0]
            char_ = gram[1]
            if not self.context_dict.get(context):  # check if context exists?
                self.context_dict[context] = {}
            try:
                self.context_dict[context][char_] += 1
            except KeyError:  # check if char for the context exists?
                self.context_dict[context][char_] = 1

    def prob(self, context, char):
        """Returns the probability of char appearing after context """
        if context not in self.context_dict:
            return 1/len(self.vocab_set)

        char_count = self.context_dict[context].get(char, 0) + self.k
        context_count = sum(self.context_dict[context].values()) + self.k * len(self.vocab_set)

        return char_count/context_count

    def random_char(self, context):
        ''' Returns a random character based on the given context and the
            n-grams learned by this model '''
        r = random.random()
        prob_cum_left = 0  # to maintain cumulative probability for vocab char
        if not self.is_vocab_sorted:
            self.vocab_set = sorted(self.vocab_set)
            self.is_vocab_sorted = True  # to avoid sorting again and again

        for char_ in self.vocab_set:
            prob_char = self.prob(context, char_)
            if prob_char + prob_cum_left > r:
                break  # break when cumulative probability exceeds r for this index
            prob_cum_left += prob_char

        return char_

    def random_text(self, length):
        """ Returns text of the specified character length based on the
            n-grams learned by this model """
        generated_text = ""
        for index in range(length):
            context = f"{start_pad(self.n-index)}{generated_text[max(0, index-self.n):index]}"  # update the context
            # for current char
            generated_text = f"{generated_text}{self.random_char(context)}"
        return generated_text

    def perplexity(self, text):
        """ Returns the perplexity of text based on the n-grams learned by
            this model """
        log_prob = 0
        for index in range(len(text)):
            context = f"{start_pad(self.n-index)}{text[max(0, index-self.n):index]}"  # update the context
            # for current char
            char_ = text[index]
            prob = self.prob(context, char_)  # compute prob for context and char
            if prob == 0:
                log_prob += float('-inf')
                break  # breaking as the overall prob is going to be 0.0
            else:
                log_prob += math.log(prob)
        if log_prob == float('-inf'):
            return float('inf')
        return math.exp(log_prob * -1/len(text))


################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        super().__init__(n, k)
        self.lambdas = None

    def get_vocab(self):
        return super().get_vocab()

    def update(self, text):
        super().update(text)
        for n in range(self.n):
            super().populate_context_dict(n, text)  # also add in context from lower order n-grams

    def prob(self, context, char, lambdas=None):  # customize to pass different lambdas in here for comparision
        if not lambdas:
            lambdas = self.lambdas # default to model default
        prob_ = 0
        for index in range(self.n + 1):
            context_ = context[self.n - index:self.n]
            prob_ += lambdas[index] * super().prob(context_, char)  # weighted probability by lambdas
        return prob_

    def compute_perplexity_with_varying_lambdas(self, text, lambdas=None):  # compute perplexity with varying lambdas
        log_prob = 0
        for index in range(len(text)):
            context = f"{start_pad(self.n-index)}{text[max(0, index-self.n):index]}"
            char_ = text[index]
            prob = self.prob(context, char_, lambdas)
            if prob == 0:
                log_prob += float('-inf')
                break  # breaking as the overall prob is going to be 0.0
            else:
                log_prob += math.log(prob)

        if log_prob == float('-inf'):
            return float('inf')

        return math.exp(log_prob * -1/len(text))

    @property
    def lambdas(self):
        return self._lambdas

    @lambdas.setter
    def lambdas(self, lambda_vals):  # call in program: model.lambdas = [<list of new lambdas>]
        """
        Setter for model lambdas
        """
        if not lambda_vals:
            self._lambdas = []
            for i in range(self.n + 1):
                self._lambdas.append(1 / (self.n + 1))
        else:
            self._lambdas = lambda_vals.copy()

    def lambdas_optimizer_on_held_out_text(self, text, max_iter=100):  # If there is no held out text, this can be
        # called on the test set and the lambdas yielding highest interpolation probability would be selected.
        """
        :param text: Text over whcih to optimize
        :param max_iter: Number of iterations to consider
        """
        min_perp = self.compute_perplexity_with_varying_lambdas(text)  # minimum perplexity to that of model default
        max_lambdas = self.lambdas
        for i in range(max_iter):
            lambdas_norm = self.random_lamdas_generator()
            perp_ = self.compute_perplexity_with_varying_lambdas(text, lambdas_norm)
            if perp_ < min_perp:
                min_perp = perp_
                max_lambdas = lambdas_norm  # update the optimized lambdas for a smaller perplexity obtained
        return max_lambdas

    def random_lamdas_generator(self):
        """
        Generates random weights
        :return:
        """
        lambdas = sorted([random.random() for _ in range(self.n + 1)], reverse=True)
        sum_lambdas = sum(lambdas)
        lambdas_norm = list(map(lambda x: x / sum_lambdas, lambdas))  # normalize in order to make sum to 1
        return lambdas_norm


################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################


if __name__ == '__main__':
    print("ngrams for n=1 and text abc:")
    print(ngrams(1, 'abc'))
    print("\nngrams for n=2 and text abc:")
    print(ngrams(2, 'abc'))

    print("\n-----Ngrammodel with n=1 and k=0--------")
    m = NgramModel(1, 0)
    print("Inserting abab")
    m.update('abab')
    print(f"Vocab now: {m.get_vocab()}")
    print("Inserting abcd")
    m.update('abcd')
    print(f"Vocab now: {m.get_vocab()}\n")

    print("Probabilities..")
    print(f"Prob of 'b' given 'a' as context: {m.prob('a', 'b')}")
    print(f"Prob of 'c' given '~' as context: {m.prob('~', 'c')}")
    print(f"Prob of 'c' given 'b' as context: {m.prob('b', 'c')}")

    print("\n------Running random char example--------")
    m = NgramModel(0, 0)
    m.update('abab')
    m.update('abcd')
    random.seed(1)
    print("25 random chars as generated by model:")
    print(f"{[m.random_char('') for i in range(25)]}")

    print("\n------Running random text example--------")
    m = NgramModel(1, 0)
    m.update('abab')
    m.update('abcd')
    random.seed(1)
    print("Generated random text of length 25:")
    print(f"{m.random_text(25)}")

    print("\n------Training on Shakespeare input file--------")

    input_text = ''
    sonnets = ''
    nytimes = ''

    with open('shakespeare_input.txt', encoding='utf-8', errors='ignore') as f:
        input_text = f.read()
        f.close()

    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        sonnets = f.read()
        f.close()

    with open('nytimes_article.txt', encoding='utf-8', errors='ignore') as f:
        nytimes = f.read()
        f.close()

    print("Training with n=2 and k=0:")
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 2)
    print("Generated random text:")
    random.seed(1)
    print(f"{m.random_text(250)}")
    print(f"\nPerplexity of model for word 'First': {m.perplexity('First')}")
    print(f"\nPerplexity of model on training set': {m.perplexity(input_text)}")
    print(f"\nPerplexity of model on sonnets': {m.perplexity(sonnets)}")
    print(f"\nPerplexity of model on nytimes article': {m.perplexity(nytimes)}")

    print("\nTraining with n=3 and k=0:")
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 3)
    print("Generated random text:")
    random.seed(1)
    print(f"{m.random_text(250)}")
    print(f"\nPerplexity of model for word 'First': {m.perplexity('First')}")
    print(f"\nPerplexity of model on training set': {m.perplexity(input_text)}")
    print(f"\nPerplexity of model on sonnets': {m.perplexity(sonnets)}")
    print(f"\nPerplexity of model on nytimes article': {m.perplexity(nytimes)}")

    print("\nTraining with n=4 and k=0:")
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 4)
    print("Generated random text:")
    random.seed(1)
    print(f"{m.random_text(250)}")
    print(f"\nPerplexity of model for word 'First': {m.perplexity('First')}")
    print(f"\nPerplexity of model on training set': {m.perplexity(input_text)}")
    print(f"\nPerplexity of model on sonnets': {m.perplexity(sonnets)}")
    print(f"\nPerplexity of model on nytimes article': {m.perplexity(nytimes)}")

    print("\nTraining with n=7 and k=0:")
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 7)
    print("Generated random text:")
    random.seed(1)
    print(f"{m.random_text(250)}")
    print(f"\nPerplexity of model for word 'First': {m.perplexity('First')}")
    print(f"\nPerplexity of model on training set': {m.perplexity(input_text)}")
    print(f"\nPerplexity of model on sonnets': {m.perplexity(sonnets)}")
    print(f"\nPerplexity of model on nytimes article': {m.perplexity(nytimes)}")

    print("\n------Comparing across different smoothing methods--------")
    print("Varying K first")

    print("\nTraining with n=7 and k=1:")
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 7, 1)
    print(f"\nPerplexity of model on training set': {m.perplexity(input_text)}")
    print(f"\nPerplexity of model on sonnets': {m.perplexity(sonnets)}")
    print(f"\nPerplexity of model on nytimes article': {m.perplexity(nytimes)}")

    print("\nTraining with n=7 and k=2:")
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 7, 2)
    print(f"\nPerplexity of model on training set': {m.perplexity(input_text)}")
    print(f"\nPerplexity of model on sonnets': {m.perplexity(sonnets)}")
    print(f"\nPerplexity of model on nytimes article': {m.perplexity(nytimes)}")

    print("\nTraining with n=7 and k=3:")
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 7, 3)
    print(f"\nPerplexity of model on training set': {m.perplexity(input_text)}")
    print(f"\nPerplexity of model on sonnets': {m.perplexity(sonnets)}")
    print(f"\nPerplexity of model on nytimes article': {m.perplexity(nytimes)}")

    print("Varying lambdas for Interpolation model")
    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 7, 1)
    print(f"Running model with lambdas: {m.lambdas}")
    print(f"\nPerplexity of model on sonnets': {m.perplexity(sonnets)}")
    print(f"\nPerplexity of model on nytimes article': {m.perplexity(nytimes)}")
    # print(f"\nPerplexity of model on training set': {m.perplexity(input_text)}")

    random.seed(1)
    random_lambdas = m.random_lamdas_generator()
    print(f"Running model with lambdas: {random_lambdas}")
    print(f"\nPerplexity of model on sonnets': {m.compute_perplexity_with_varying_lambdas(sonnets, random_lambdas)}")
    print(f"\nPerplexity of model on nytimes article': "
          f"{m.compute_perplexity_with_varying_lambdas(nytimes, random_lambdas)}")

    random_lambdas = m.random_lamdas_generator()
    print(f"\n---Running model with lambdas: {random_lambdas}---")
    print(f"\nPerplexity of model on sonnets': {m.compute_perplexity_with_varying_lambdas(sonnets, random_lambdas)}")
    print(f"\nPerplexity of model on nytimes article': "
          f"{m.compute_perplexity_with_varying_lambdas(nytimes, random_lambdas)}")

    random.seed(1)
    print("\n---Running with optimized lambdas over sonnets---")
    optimized_lambdas = m.lambdas_optimizer_on_held_out_text(sonnets)
    print(f"Running model with optimized lambdas: {optimized_lambdas}")
    print(f"\nPerplexity of model on sonnets': {m.compute_perplexity_with_varying_lambdas(sonnets, optimized_lambdas)}")
    print(f"\nPerplexity of model on nytimes article': "
          f"{m.compute_perplexity_with_varying_lambdas(nytimes, optimized_lambdas)}")
