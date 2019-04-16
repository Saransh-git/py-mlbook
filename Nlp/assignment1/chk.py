import matplotlib.pyplot as plt

pp_pos = [(0.83821748607411, 0.83064),
 (0.8470283373735685, 0.81064),
 (0.8527018926094031, 0.79656),
 (0.8563564875491481, 0.78408),
 (0.8594459784448205, 0.77192),
 (0.8637973304276764, 0.76104),
 (0.8682790697674418, 0.74672),
 (0.8733825361832647, 0.72896),
 (0.8794902669074854, 0.7012)]

precision = []
recall = []

for item in pp_pos:
    precision.append(item[0])
    recall.append(item[1])

plt.figure()
axs = plt.gca()
axs.plot(recall, precision, 'o--')
axs.set_title("Precision vs Recall +ve label")
axs.set_xlabel("Recall")
axs.set_ylabel("Precision")
plt.show()

pp_neg = [(0.8321572980258464, 0.83968),
 (0.8184398251131395, 0.8536),
 (0.8091270734819485, 0.8624),
 (0.800885282183696, 0.86848),
 (0.7930007986640528, 0.87376),
 (0.7864445556588261, 0.88),
 (0.7778245614035088, 0.88672),
 (0.7674195098510331, 0.89432),
 (0.7515631235865372, 0.90392)]

precision = []
recall = []

for item in pp_neg:
    precision.append(item[0])
    recall.append(item[1])

plt.figure()
axs = plt.gca()
axs.plot(recall, precision, 'o--')
axs.set_title("Precision vs Recall -ve label")
axs.set_xlabel("Recall")
axs.set_ylabel("Precision")
plt.show()
