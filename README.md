# 🧠 miniAZ-R: A Tiny Reinforced Math Solver Inspired by AZR (2025)

Inspired by the [AZ-R paper](https://paperswithcode.com/paper/absolute-zero-reinforced-self-play-reasoning)
“Absolute Zero: Reinforced Self-Play Reasoning with Zero Data” (2025) by Andrew Zhao Yiran Wu, Tong Wu, Quentin Xu, and collaborators.

This repository is not affiliated with the original authors, but serves as a practical miniature replication and experimental foundation toward that vision.

<p align="center">
  <img src="https://github.com/parastak/MiniAZ-R-Reinforcement_Learning_Mathematical_Reasoning_solver/blob/main/resources/miniAZ_R.png" alt="Screenshot" width="600" />
</p>

🎯 Goal
--
Train an RL agent from scratch to deduce symbolic mathematical equations through trial and error, guided by curriculum learning and self-play without supervised labels
or solvers.  From initial failure and debugging to achieving mastery on a curriculum of increasingly difficult mathematical tasks.


# 🔍 Features Implemented

✅ Actor-Critic agent with entropy & KL regularization

✅ Reward shaping and log_std tracking for exploration

✅ Curriculum learning pipeline (Easy → Medium → Hard tasks)

✅ Progressive reward & error convergence

✅ Evaluation dashboard & symbolic tagging

✅ Logging for interpretability and training insights



## 📸 Screenshots

Here are some results which showing a proper improvement in the agent reducing loss and growth in the reward signals 

# Initial agent learning results 
<p align="center">
  <img src="https://github.com/parastak/MiniAZ-R-Reinforcement_Learning_Mathematical_Reasoning_solver/blob/main/resources/training1.png" alt="Screenshot2" width="500" />
  <img src="https://github.com/parastak/MiniAZ-R-Reinforcement_Learning_Mathematical_Reasoning_solver/blob/main/resources/training%201.png" alt="Screenshot1" width="500" />
</p>


# latest agent policy results 
<p align="center">
  <img src="https://github.com/parastak/MiniAZ-R-Reinforcement_Learning_Mathematical_Reasoning_solver/blob/main/resources/Screenshot%202025-07-08%20192719.png" alt="Screenshot2" width="500" />
  <img src="https://github.com/parastak/MiniAZ-R-Reinforcement_Learning_Mathematical_Reasoning_solver/blob/main/resources/Screenshot%202025-07-08%20192731.png" alt="Screenshot1" width="500" />
  <img src="https://github.com/parastak/MiniAZ-R-Reinforcement_Learning_Mathematical_Reasoning_solver/blob/main/resources/Screenshot%202025-07-08%20192746.png" alt="Screenshot3"
 />
</p>


## 🔭 Future Work (toward full AZR)

- Add **post-hoc error correction**
- Integrate **code-based symbolic execution**
- Diversify tasks to include chain-of-thought
- Add **multi-agent curriculum evolution**
- Enable **multi-step program synthesis reasoning**

---

## 🤝 Contribution & Citation

This project is not affiliated with the original AZR authors.  
Inspired by research, built independently as a learning replica.

Open to collaborations, extensions, and feedback. Star ⭐ and fork if useful!

---
