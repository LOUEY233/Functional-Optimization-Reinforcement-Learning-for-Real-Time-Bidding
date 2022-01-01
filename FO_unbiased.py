import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from Agent import *
from config import *

device = torch.device("cpu")
bid_option = np.arange(10, 100, 1)
L = np.arange(0.00,0.01,0.00005) #need change
action_space1 = len(bid_option)
action_space2 = len(L)

h = 1
theta = 0.001 # CTR
budget = 25000
budget_consumption_rate = 0  # recent consumption rate
operation = 0
interval = 0
win_rate = 0
global_bi = []
global_wi = []
global_zi = []

#baseline
Agent1_baseline1 = DQNAgent(budget=budget, state=[0, 0, 0, 0], observation_space=4, action_space=action_space1)
#state
Agent2_operation1 = DQNAgent(budget=budget, state=[0, 0, 0, 0, 0], observation_space=5, action_space=action_space1)
#action
Agent3_operation3 = DQNAgent(budget=budget, state=[0, 0, 0, 0], observation_space=4, action_space=action_space2)
#reward
Agent4_operation2 = DQNAgent(budget=budget, state=[0, 0, 0, 0], observation_space=4,action_space=action_space1)
Agents = [Agent1_baseline1,Agent2_operation1,Agent3_operation3,Agent4_operation2]

total_request = 0
writer = pd.ExcelWriter("MARL.xlsx")

win_record = [0,0,0,0]

for time in range(10):
    step = 1  # 1-1000
    total_market = []


    print(time)
    for i in range(4):
        Agents[i].setup()
        Agents[i].budget = budget
        if i == 1:
            Agents[i].state = [budget, budget_consumption_rate, win_rate, step,0]
        else:
            Agents[i].state = [budget, budget_consumption_rate, win_rate, step]
        Agents[i].next_state = Agents[i].state
        Agents[i].replayBuffer = ReplayBuffer(1000)

# checkpoint = torch.load("./pth/weight_ipinyou_ddqn_cpc.pt")
# Our_client.network.load_state_dict(checkpoint)
    n_request_left = 1000
    bid_p = []
    unique_bid = list()
    win_prob = list()

    for request in range(1, n_request_left + 1):
        epsilon = epsilon_by_frame(total_request)
        done = 0
        bid_p = []
        for i in range(4):
            Agents[i].action = Agents[i].network.act(Agents[i].state, epsilon)
            if i == 2: # for agent which take lamda as action
                if random.random() > epsilon:
                    l = L[Agents[i].action] # l = lambda / c
                    temp_price = Agents[i].get_price_unbiased(unique_bid,win_prob,l) # For agent3
                else:
                    temp_price = random.randrange(10,100,1)
                temp_price = int(temp_price)
            else:
                temp_price = bid_option[Agents[i].action]
            if Agents[i].budget < temp_price:
                temp_price = Agents[i].budget
            bid_p.append(temp_price)
            global_bi.append(temp_price)
            Agents[i].bid_log.append(temp_price)
        total_market.append(np.sort(bid_p)[2])
        if len(bid_p) != 4:
            print("error")

        reward_1 = []
        for i in range(4):
            if bid_p[i] == np.max(bid_p):
                reward_1.append(5)
                if i == 1 or i == 2 or i == 3:
                    Agents[i].update_w_dw(bid_price=bid_p[i], flag=1,request=request)
                if np.sort(bid_p)[2] == 0:
                    second_price = np.sort(bid_p)[3]
                else:
                    second_price = np.sort(bid_p)[2]
                global_wi.append(1)
                global_zi.append(second_price)
                Agents[i].budget -= second_price
                Agents[i].interval.append(np.max(bid_p) - second_price)
                Agents[i].consumption.append(second_price)
                Agents[i].win_log.append(second_price)
                Agents[i].win += 1
                Agents[i].win_rate.append(1)
                Agents[i].win_period += 1
            else:
                if i == 1 or i == 2 or i == 3:
                    Agents[i].update_w_dw(bid_price=bid_p[i],flag=0,request=request)
                global_wi.append(0)
                global_zi.append(0)
                reward_1.append(-1)
                Agents[i].consumption.append(0)
                Agents[i].win_rate.append(0)
            Agents[i].budget_log.append(Agents[i].budget)
        
        # global_data = pd.DataFrame({'bi':global_bi,'wi':global_wi,'zi':global_zi})
        # print(global_data)
        # unique_bid,win_prob = p.map(win_prob_second_list, args=(global_bi,global_wi,global_zi))
        unique_bid,win_prob = win_prob_second_list(global_bi,global_wi,global_zi)
        # print(unique_bid,win_prob)


        for i in range(4):
            win_rate = np.sum(Agents[i].win_rate) / (len(Agents[i].win_rate)+0.001)
            consumption_rate = np.mean(Agents[i].consumption)
            remaining_budget = Agents[i].budget
            Agents[i].next_state[3] = Agents[i].state[3] + 1
            Agents[i].next_state[2] = win_rate
            Agents[i].next_state[1] = consumption_rate
            Agents[i].next_state[0] = remaining_budget
            if len(Agents[i].state) == 5:  # for agent 2, using unbiased model
                Agents[i].next_state[4] = Agents[i].get_lambda_unbiased(unique_bid,win_prob,bid_p[i],theta)

        for i in range(4):
            if Agents[0].budget < 100 and Agents[1].budget < 100 and Agents[2].budget < 100 and Agents[3].budget < 100:
                compare_win = []
                for j in range(4):
                    compare_win.append(Agents[j].win)
                if i == np.argmax(compare_win):
                    Agents[i].reward = reward_1[i] + 200
                else:
                    Agents[i].reward = reward_1[i]
            else:
                if i == 3:
                    # modify
                    # print(f"reward: {reward_1[i]}")
                    # print(f"lambda: {500*Agents[i].get_lambda(unique_bid,win_prob,bid_p[3],theta)}")
                    Agents[i].reward = reward_1[i] + 500*Agents[i].get_lambda_unbiased(unique_bid,win_prob,bid_p[3],theta)
                    if request % 1000 == 0:
                        print(500*Agents[i].get_lambda_unbiased(unique_bid,win_prob,bid_p[3],theta))
                Agents[i].reward = reward_1[i]

        for i in range(4): 
            Agents[i].replayBuffer.push(Agents[i].state, Agents[i].action, Agents[i].reward, Agents[i].next_state, done)
            Agents[i].state = Agents[i].next_state
            Agents[i].episode_reward += Agents[i].reward

        if request % batch_size == 0:
            for i in range(4):
                loss = compute_td_loss(Agents[i].network, Agents[i].optimizer, Agents[i].replayBuffer, gamma, batch_size)
            if Agents[0].budget < 100 and Agents[1].budget < 100 and Agents[2].budget < 100 and Agents[3].budget < 100:
                break

        if request % 200 == 0:
            print("state", Agents[i].state)
            for i in range(4):
                print(Agents[i].reward)
            print(bid_p)
            print("*" * 100)

        # for i in range(4):
        #     if Agents[i].state[4] == 2000:
        #         # Our_client.budget = 40000000
        #         Agents[i].state[1] = 0
        #         Agents[i].state[2] = 0
        #         Agents[i].state[3] = 0
        #         Agents[i].state[4] = 0
        #         Agents[i].interval = []
        #         Agents[i].consumption = []
        #         Agents[i].win_rate = []
        #         Agents[i].total_win.append(Agents[i].win_log[-2000:])
    for ii in range(4):
        print(Agents[ii].win)
    print("*****************")
    output = {"market_price": total_market, "bid_price1": Agents[0].bid_log,
              "bid_price2": Agents[1].bid_log, "bid_price3": Agents[2].bid_log,
              "bid_price4": Agents[3].bid_log,
              "A1_b_log":Agents[0].budget_log,"A2_b_log":Agents[1].budget_log,"A3_b_log":Agents[2].budget_log,"A4_b_log":Agents[3].budget_log}
    output = pd.DataFrame(output)
    output.to_excel(writer,sheet_name='{}'.format(time))

    total_request += request
    print(epsilon)
    for i in range(4):
        print(Agents[i].budget)
for i in range(4):
    torch.save(Agents[i].network.state_dict(), "./agent{}.pt".format(i))
    # ttotal_win.append(Agents[i].win_log)
writer.save()

# print("train win:",len(ttotal_win[0]),"total request:",len(data),"win percent:",len(ttotal_win[0])/len(data))

# # win price distribution (our client)
# for i in range(len(Our_client.total_win)):
#     plt.hist(Our_client.total_win[i], bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
#     plt.hist(data,bins=40,facecolor="green",edgecolor="black",alpha=0.7)
# plt.show()
