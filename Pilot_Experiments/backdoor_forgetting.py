import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

"""
Function (Figure 1): This script plots the Attack Success Rate (ASR) and Accuracy (ACC) over 5 downstream fine-tuning epochs for the AddSent, Sleeper,
and VPI attack models. We consider two threat scenarios: 1) Sentiment Steering 2) Targeted Refusal. Downstream task: SST-2.

Developer: Xingyi Zhao. 
Update: 2026-06-24
Utah, Logan, USA
"""

parser_backdoor_forgetting = argparse.ArgumentParser(description="Plot ASR and ACC for AddSent, Sleeper, and VPI attack models.")
parser_backdoor_forgetting.add_argument("--threat_scenario", type=str, default="sentiment_steering", help="Threat scenario: sentiment_steering or targeted_refusal. ")
parser_backdoor_forgetting.add_argument("--attack_model", type=str, default="AddSent", help="Attack model: AddSent, Sleeper, or VPI.")
args_backdoor_forgetting = parser_backdoor_forgetting.parse_args()

# ASR and ACC values for the attack model
with open(f"./Data/asr_{args_backdoor_forgetting.threat_scenario}_{args_backdoor_forgetting.attack_model.lower()}.txt", "r") as f:
    list_sentiment_analysis_asr = [float(line.strip()) for line in f.readlines()]

with open(f"./Data/acc_{args_backdoor_forgetting.threat_scenario}_{args_backdoor_forgetting.attack_model.lower()}.txt", "r") as f:
    list_sentiment_analysis_acc = [float(line.strip()) for line in f.readlines()]

epochs_sentiment_analysis = [0, 1, 2, 3, 4, 5]

# Plot the curve
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
ax.plot(epochs_sentiment_analysis, list_sentiment_analysis_asr, linewidth=7, marker='o', markersize=20, label="ASR", color=(0.5, 0.2, 0.8))
ax.plot(epochs_sentiment_analysis, list_sentiment_analysis_acc, linewidth=7, marker='^', markersize=20, label="ACC", color=(0.5, 0, 0))

ax.spines['left'].set_linewidth(7)
ax.spines['bottom'].set_linewidth(7)
ax.spines['right'].set_linewidth(7)
ax.spines['top'].set_linewidth(7)

ax.set_xticks(epochs_sentiment_analysis)
ax.tick_params(axis='both', labelsize=55, width=7, length=10)
ax.set_title(args_backdoor_forgetting.attack_model, fontsize=55, pad=18)

ax.set_ylim(-10, 110)
ax.yaxis.set_major_locator(MultipleLocator(20))

plt.xlabel("Epoch", fontsize=55)
plt.savefig(f"{args_backdoor_forgetting.threat_scenario}_{args_backdoor_forgetting.attack_model.lower()}.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()