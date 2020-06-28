import torch
from typing import NamedTuple
import numpy as np
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateMRTA(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc (coordinates of all the locations including the depot)
    distance_matrix: torch.Tensor # distance matrix for all the coordinates
    time_matrix: torch.Tensor # time matrix between all the coordinates using the speed of the agents
    deadline: torch.Tensor # deadline for all the tasks (special case for the depot, keep a very large time)
    # demand: torch.Tensor # we do not need this, we can remove this or set the demand as 1 or something equal to the quantity for 1 time delivery

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows (this is basically the ids for all the location, which are considered as integers)



    # robot specific
    robots_initial_decision_sequence: torch.Tensor # for timestep 1, all robots need decision, so we set a sequence for this
    robots_task_done_success: torch.Tensor # for each robot, this variable tracks the id of the task done successfully
    robots_task_missed_deadline: torch.Tensor # for each robot, this variable tracks the id of the task with missed deadline
    robots_task_visited: torch.Tensor # keeps track of all the nodes visited by all the robots (successful or not)
    robots_distance_travelled: torch.Tensor # keeps track of the total distance travelled by the robots (this is updated everytime a new decision is made and also during the end of the simulation)
    # robots_total_tasks_done # optional
    robots_next_decision_time: torch.Tensor # tracks the next decision time for all the robots
    robots_range_remaining: torch.Tensor # tracks of the range remaining for all the robots
    robots_capacity: torch.Tensor # keeps track of the capacity of the robot
    robots_current_destination: torch.Tensor
    robots_start_point: torch.Tensor
    robot_taking_decision_range: torch.Tensor


    #general - frequent changing variable
    current_time: torch.Tensor # stores the value for the current time
    robot_taking_decision: torch.Tensor  # stores the id of the robot which will take the next decision
    next_decision_time: torch.Tensor # time at which the next decision is made. (0 t begin with)
    previous_decision_time: torch.Tensor # time at which the previous decision was made


    # for performance tracking
    tasks_done_success: torch.Tensor # keeps track of all the task id which was done successfully
    tasks_missed_deadline: torch.Tensor # keeps track of all the tasks which are visited but the deadline was missed
    tasks_visited: torch.Tensor # keeps track of all the tasks which are visited (successful or not)
    depot: torch.Tensor



    #end
    is_done: torch.Tensor


    # State
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    # VEHICLE_CAPACITY = 1.0  # Hardcoded
    n_nodes = 40
    n_agents = 4
    max_range = 2
    max_capacity = 5
    max_speed = 10


    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_[:,:,1:]
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                prev_a=self.prev_a[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key],
            )
        return super(StateMRTA, self).__getitem__(key)

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        depot = input['depot']
        loc = input['loc']
        n_agents = 4
        max_range = 2
        max_capacity = 5
        max_speed = 10
        coords = torch.cat((depot[:, None, :], loc), -2)
        distance_matrix = (coords[:, :, None, :] - coords[:, None, :, :]).norm(p=2, dim=-1)
        time_matrix = torch.mul(distance_matrix, (1/max_speed))
        deadline = input['deadline']
        # n_nodes = torch.tensor(loc.size()[1], dtype=torch.uint8)

        batch_size, n_loc, _ = loc.size()
        return StateMRTA(
            coords=coords,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=input['depot'][:, None, :],  # Add step dimension
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            robots_initial_decision_sequence = torch.from_numpy(np.arange(0, n_agents)),#torch.from_numpy(np.arange(0, n_agents)),
            robots_task_done_success = torch.zeros((batch_size, n_agents), dtype=torch.uint8),
            robots_task_missed_deadline = torch.zeros((batch_size, n_agents), dtype=torch.uint8),
            robots_task_visited = torch.zeros((batch_size, n_agents), dtype=torch.uint8),
            robots_distance_travelled  = torch.zeros((batch_size, n_agents), dtype=torch.float),
            robots_next_decision_time =  torch.zeros((batch_size, n_agents), dtype=torch.float),
            robots_range_remaining = torch.mul(torch.ones((batch_size,n_agents), dtype=torch.float), max_range),
            robots_capacity = torch.mul(torch.ones((batch_size,n_agents), dtype=torch.float), max_capacity),
            current_time = torch.zeros((batch_size, 1), dtype=torch.float),
            robot_taking_decision = torch.zeros((batch_size, 1), dtype=torch.uint8),
            next_decision_time = torch.zeros((batch_size, 1), dtype=torch.float),
            previous_decision_time = torch.zeros((batch_size, 1), dtype=torch.float),
            tasks_done_success = torch.zeros((batch_size, 1), dtype=torch.uint8),
            tasks_missed_deadline = torch.zeros((batch_size, 1), dtype=torch.uint8),
            tasks_visited = torch.zeros((batch_size, 1), dtype=torch.uint8),
            is_done = torch.zeros((batch_size, 1), dtype=torch.uint8),
            distance_matrix = distance_matrix,
            time_matrix = time_matrix,
            deadline = deadline,
            robots_current_destination = torch.zeros((batch_size, n_agents), dtype=torch.uint8),
            robots_start_point = torch.zeros((batch_size, n_agents), dtype=torch.uint8),
            robot_taking_decision_range = torch.mul(torch.ones(batch_size, 1, dtype=torch.float), max_range),
            depot = torch.zeros((batch_size, 1), dtype=torch.uint8)
            # n_nodes = n_nodes
        )

    def get_final_cost(self):

        assert self.all_finished()

        len = self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)
        # torch.mul(len, self.)
        return len

    def update(self, selected):
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected

        previous_time = self.current_time

        current_time = self.next_decision_time


        #update mileage
        robots_range_remaining = self.robots_range_remaining
        robot_taking_decision = self.robot_taking_decision

        # print('Current time: ', current_time[0].item())
        # print("Agent taking decision: ", self.robot_taking_decision[0].item())
        # print('Agent decision taking time: ', self.robots_next_decision_time[0, self.robot_taking_decision[0].item()].item())
        # print('Envt decision time: ', self.next_decision_time[0].item())
        # print('Previous decision time: ', self.previous_decision_time[0].item())
        # print("Agent range remaining: ", robots_range_remaining[0, robot_taking_decision[0].item()].item())
        # print('Agent start point: ', self.robots_start_point[0, robot_taking_decision[0].item()].item())
        # print('Agent current destination: ', self.robots_current_destination[0, robot_taking_decision[0].item()].item())
        # print('Agent current capacity', self.robots_capacity[0, robot_taking_decision[0].item()].item())
        # print('Current node deadline: ', self.deadline[0, self.robots_current_destination[0, robot_taking_decision[0].item()].item() -1].item())
        # print('Selected action:', selected[0].item())
        # new_dist = self.distance_matrix[0, self.robots_current_destination[0, robot_taking_decision[0].item()].item(), selected[0].item()].item()
        # print('Distance to selected node: ', new_dist)
        # print('Robot next decision taking time: ',self.robots_next_decision_time[0, self.robot_taking_decision[0].item()] + self.time_matrix[
        #         0, self.robots_current_destination[0, robot_taking_decision[0].item()].item(), selected[
        #             0].item()].item())
        # print('Tasks visited: ', self.tasks_visited[0].item())

        # print(selected)

        cur_coords = self.coords[self.ids, self.robots_current_destination[self.ids, self.robot_taking_decision.to(dtype=torch.int64)].to(dtype=torch.int64)]

        # update the capacity if its before the deadline
        for id in self.ids:

            time = self.time_matrix[
                id, self.robots_current_destination[id, robot_taking_decision[id].item()].item(), selected[
                    id].item()].item()

            self.robots_next_decision_time[id, self.robot_taking_decision[id].item()] += time

            # self.robots_distance_travelled[id, self.robot_taking_decision[id].item()] += self.distance_matrix[
            #     id, self.robots_start_point[id, robot_taking_decision[id].item()].item(),
            #     self.robots_current_destination[id, robot_taking_decision[id].item()].item()].item()
            self.robots_distance_travelled[id, self.robot_taking_decision[id].item()] += self.distance_matrix[
                id, self.robots_current_destination[id, robot_taking_decision[id].item()].item(), selected[id].item()].item()

            # if self.robots_current_destination[id, robot_taking_decision[id].item()].item() == 0:
            if selected[id].item() == 0:
                self.robots_capacity[id, self.robot_taking_decision[id].item()] = self.max_capacity
                robots_range_remaining[id, robot_taking_decision[id].item()] = self.max_range
            else:
                robots_range_remaining[id, robot_taking_decision[id].item()] -= self.distance_matrix[
                    id, self.robots_current_destination[id, robot_taking_decision[id].item()].item(), selected[
                        id].item()].item()

                if self.deadline[id, selected[id].item() -1].item() > self.robots_next_decision_time[id, self.robot_taking_decision[id].item()]:
                    self.robots_capacity[id, self.robot_taking_decision[id].item()] -= 1
                    self.tasks_done_success[id] += 1

                else:
                    self.tasks_missed_deadline[id] += 1

                self.tasks_visited[id] += 1


            # print('Robot new range: ', robots_range_remaining[id, robot_taking_decision[id].item()].item())
            self.robots_start_point[id, self.robot_taking_decision[id].item()] = self.robots_current_destination[id, self.robot_taking_decision[id].item()].item()
            # print('Robot new start point: ', self.robots_start_point[id, self.robot_taking_decision[id].item()].item())
            self.robots_current_destination[id, self.robot_taking_decision[id].item()] = selected[id].item()
            # print('Robot new destination: ', self.robots_current_destination[id, self.robot_taking_decision[id].item()].item())
            # print('New decision times: ', self.robots_next_decision_time)

        sorted_time, indices = torch.sort(self.robots_next_decision_time)
        # print('Next decision time: ', sorted_time[0,0].item())
        # print('Agent taking next decision: ', indices[0,0].item())
        robot_taking_decision_range = self.robot_taking_decision_range
        for id in self.ids:
            robot_taking_decision_range[id] = robots_range_remaining[id, indices[id, 0].item()].item()
        # print('Range of robot taking next decision: ',robot_taking_decision_range[0].item())
        # print('All ranges: ', robots_range_remaining[0])

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)

        new_cur_coord = self.coords[self.ids, selected]
        # cur_coord = self.coords.gather(
        #     1,
        #     selected[:, None].expand(selected.size(0), 1, self.coords.size(-1))
        # )[:, 0, :]
        lengths = self.lengths + (new_cur_coord - cur_coords).norm(p=2, dim=-1)
        # print('Total length: ', lengths[0].item())
        visited_[:,:,0] = 0


        # print('visited: ', visited_[0])
        # print('***************** End of decision making process*******')
        # if end
        return self._replace(
            prev_a=prev_a, previous_decision_time = previous_time, current_time = current_time,
            robots_range_remaining = robots_range_remaining, robot_taking_decision = indices[self.ids,0],
            next_decision_time = sorted_time[self.ids,0],
            robot_taking_decision_range = robot_taking_decision_range,
            visited_=visited_,
            lengths=lengths, cur_coord=new_cur_coord,
            i=self.i + 1
        )


    def all_finished(self):
        # return self.i.item() >= self.demand.size(-1) and self.visited.all()
        return self.visited.all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """

        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]
        else:
            visited_loc = mask_long2bool(self.visited_)

        mask_loc = visited_loc.to(torch.bool) #| exceeds_cap

        robot_taking_decision = self.robot_taking_decision.tolist()

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (self.robots_current_destination[self.ids,robot_taking_decision] == 0) & ((mask_loc == 0).int().sum(-1) > 0)
        full_mask = torch.cat((mask_depot[:, :, None], mask_loc), -1)
        for id in self.ids:
            robot_taking_decision = self.robot_taking_decision
            capacity = self.robots_capacity[id, robot_taking_decision[id].item()].item()
            if capacity < 1:
                full_mask[id, 1:, :] = True
            else:
                avail_range = self.robots_range_remaining[id, robot_taking_decision[id].item()].item()
                robot_dest = self.robots_current_destination[id, robot_taking_decision[id].item()].item()
                if robot_dest != 0:
                    for i in range(self.n_nodes):
                        d1 = self.distance_matrix[id, robot_dest, i+1].item()
                        d2 = self.distance_matrix[id, i+1, 0].item()
                        if not full_mask[id,:,i+1].item() and avail_range < d1+d2:
                            full_mask[id, :, i + 1] = True

        return full_mask

    def construct_solutions(self, actions):
        return actions
