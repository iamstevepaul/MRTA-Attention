import pickle
import torch
import numpy as np
import random

if __name__ == "__main__":
    n_samples = 10000
    max_n_agent = 10

    n_agents_available = torch.tensor([2,3,5,7])

    agents_ids = torch.randint(0, 4, (n_samples, 1))

    groups = torch.randint(1, 3, (n_samples, 1))

    max_range = 4
    max_capacity = 10
    max_speed = .01

    dist = torch.randint(1, 5, (n_samples, 1))

    data = []

    n_tasks = 100

    for i in range(n_samples):
        n_agents = n_agents_available[agents_ids[i, 0].item()].item()
        agents_location = (torch.randint(0, 101, (max_n_agent, 2)).to(torch.float) / 100)

        loc = torch.FloatTensor(n_tasks, 2).uniform_(0, 1)
        workload = torch.FloatTensor(n_tasks).uniform_(.2, .2)
        d_low = (((loc[:, None, :].expand((n_tasks, max_n_agent, 2)) - agents_location[None].expand((n_tasks, max_n_agent, 2))).norm(2, -1).max()/max_speed) + 20).to(torch.int64) + 1
        d_high = ((35)*(45)*100/(380) + d_low).to(torch.int64) + 1
        d_low = d_low*(.5*groups[i, 0])
        d_high = ((d_high * (.5 * groups[i, 0])/10).to(torch.int64) + 1)*10
        deadline_normal = (torch.rand(n_tasks, 1) * (d_high - d_low) + d_low).to(torch.int64) + 1

        n_norm_tasks = dist[i, 0]*25
        rand_mat = torch.rand(n_tasks, 1)
        k = n_norm_tasks.item()  # For the general case change 0.25 to the percentage you need
        k_th_quant = torch.topk(rand_mat.T, k, largest=False)[0][:, -1:]
        bool_tensor = rand_mat <= k_th_quant
        normal_dist_tasks = torch.where(bool_tensor, torch.tensor(1), torch.tensor(0))

        slack_tasks = (normal_dist_tasks - 1).to(torch.bool).to(torch.int64)

        normal_dist_tasks_deadline = normal_dist_tasks*deadline_normal

        slack_tasks_deadline = slack_tasks*d_high

        deadline_final = normal_dist_tasks_deadline + slack_tasks_deadline

        case_info = {
                    'loc': loc,
                    'depot': torch.FloatTensor(1,2).uniform_(0, 1),
                    'deadline':deadline_final.to(torch.float).view(-1),
                    'workload': workload,
                    'initial_size':100,
                    'n_agents': torch.tensor([[n_agents]]),
                    'max_range':max_range,
                    'max_capacity':max_capacity,
                    'max_speed':max_speed,
                    'enable_capacity_constraint':False,
                    'enable_range_constraint':False,
                }

        data.append(case_info)


    dt = 0
    pickle.dump(data, open("mrta_gini_validation_init.pkl", 'wb'))