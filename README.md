## Overview

This is the public code repository for our work
[On Continuous Monitoring of Risk Violations under Unknown Shift](https://alextimans.github.io/) presented at [UAI 2025](https://www.auai.org/uai2025/).


#### Abstract :memo:
---

Machine learning systems deployed in the real world must operate under dynamic and often unpredictable distribution shifts. This challenges the validity of statistical safety assurances on the system's risk established beforehand. Common risk control frameworks rely on fixed assumptions and lack mechanisms to continuously monitor deployment reliability. In this work, we propose a general framework for the real-time monitoring of risk violations in evolving data streams. Leveraging the `testing by betting' paradigm, we propose a sequential hypothesis testing procedure to detect violations of bounded risks associated with the model's decision-making mechanism, while ensuring control on the false alarm rate. Our method operates under minimal assumptions on the nature of encountered shifts, rendering it broadly applicable. We illustrate the effectiveness of our approach by monitoring risks in outlier detection and set prediction under a variety of shifts.

---

#### Citation
If you find this repository useful, please consider citing our work:

```
@inproceedings{timans2025riskmonitor,
    title = {On Continuous Monitoring of Risk Violations under Unknown Shift}, 
    author = {Alexander Timans and Rajeev Verma and Eric Nalisnick and Christian Naesseth},
    booktitle = {Proceedings of the 41st Conference on Uncertainty in Artificial Intelligence},
    year = {2025}
}
```

#### Acknowledgements
The [Robert Bosch GmbH](https://www.bosch.com) is acknowledged for financial support.

## Repo structure
Experiment code and instructions will be updated here shortly, please bear with us!

#### Still open questions?

If there are any problems you encounter which have not been addressed, please feel free to create an issue or reach out! 
