# Project Title: Reinforcement Learning-Based Adaptive Load Balancing in Cloud Environments

## Problem Statement

Cloud computing environments face highly dynamic and unpredictable workloads that challenge traditional static load balancing algorithms. Chawla et al.’s [“Reinforcement Learning- Based Adaptive Load Balancing for Dynamic Cloud Environments”](https://arxiv.org/pdf/2409.04896.pdf) demonstrated significant improvements over round-robin, least-connections, and weighted algorithms in simulation. However, its evaluation
remains limited to synthetic CloudSim data. We will:
1. Replicate Chawla et al.’s Q-learning framework in CloudSim, validating response time, resource utilization, and task completion metrics.
2. Extend the evaluation to real-world datasets (e.g., Google Cluster Trace) by retraining and testing the RL agent on production workload traces.
3. Compare the RL approach against established ML-based and heuristic techniques (DQN, PPO, genetic algorithms, ant colony optimization) using identical datasets and standardized benchmarks.(If time permits

## Datasets Used
- CloudSim Synthetic Workloads (to replicate original experiments) – https://github.com/Cloudslab/cloudsim-plus [Dataset used in original paper]
- Google Cluster Data (2011): Machine events, job events, resource usage – https://github.com/google/cluster-data
- Alibaba Cluster Trace (2018): Task scheduling, resource consumption – https://github.com/alibaba/clusterdata
- Microsoft Azure Public Dataset: VM performance counters – https://azure.microsoft.com/en-ius/services/open-datasets/azure-compute

## ML Techniques to be used
- Q-Learning (baseline replication)
- Deep Q-Network (DQN) with target networks and experience replay
- Proximal Policy Optimization (PPO) for continuous policy improvements
- Multi-Objective Reward Shaping incorporating response time, energy, and SLA penalties
- Heuristic Baselines: Genetic Algorithm, Ant Colony Optimization for comparison
- Evaluation Metrics: Average response time, CPU/memory utilization, throughput, SLA compliance, fairness index

## List of References
1. Chawla, K. “Reinforcement Learning-Based Adaptive Load Balancing for Dynamic Cloud Environments.” arXiv:2409.04896 (2024).
2. Calheiros, R. N., et al. “CloudSim: A toolkit for modeling and simulation of cloud computing environments.” Software: Practice and Experience 41 (2011):23–50.
3. H. Mao, M. Alizadeh, I. Menache, and S. Kandula, “Resource management with deep reinforcement learning,” in Proc. 15th ACM Workshop Hot Topics in Networks (HotNets), pp. 50–56, 2016.
4. Meng, X., et al. “Efficient Resource Provisioning in Data Centers with Firefly Algorithm.” IEEE Access 7 (2019):66 218–66 231.
5. Mnih, V., et al. “Human-level control through deep reinforcement learning.” Nature 518(2015):529–533

## About US:
1. Anupam Rawat (NoviceCoderInfinity)
2. Cherish Jain (CJain2004)
3. Saurabh Kumar (srbh001)
