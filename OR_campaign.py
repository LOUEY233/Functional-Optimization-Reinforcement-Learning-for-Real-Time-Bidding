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

budget = 50000
budget_consumption_rate = 0  # recent consumption rate
operation = 0
interval = 0
win_rate = 0

Agent1_baseline1 = DQNAgent(budget=budget, state=[0, 0, 0, 0], observation_space=4, action_space=action_space1)
Agent2_baseline2 = DQNAgent(budget=budget, state=[0, 0, 0, 0], observation_space=4, action_space=action_space1)
Agent3_operation1 = DQNAgent(budget=budget, state=[0, 0, 0, 0], observation_space=4, action_space=action_space2)
Agent4_operation2 = DQNAgent(budget=budget, state=[0, 0, 0, 0], observation_space=4,action_space=action_space1)
Agents = [Agent1_baseline1,Agent2_baseline2,Agent3_operation1,Agent4_operation2]

total_request = 0
writer = pd.ExcelWriter("MARL.xlsx")
for time in range(10):
    step = 1  # 1-1000
    total_market = []

    print(time)
    for i in range(4):
        Agents[i].setup()
        Agents[i].budget = budget
        Agents[i].state = [budget, budget_consumption_rate, win_rate, step]
        Agents[i].next_state = Agents[i].state
        Agents[i].replayBuffer = ReplayBuffer(1000)

# checkpoint = torch.load("./pth/weight_ipinyou_ddqn_cpc.pt")
# Our_client.network.load_state_dict(checkpoint)
    n_request_left = 5000
    bid_p = []

    for request in range(1, n_request_left + 1):
        epsilon = epsilon_by_frame(total_request)
        done = 0
        bid_p = []
        for i in range(4):
            Agents[i].action = Agents[i].network.act(Agents[i].state, epsilon)
            if i == 2:
                if random.random() > epsilon:
                    l = L[Agents[i].action]
                    temp_price = Agents[i].get_price(Agents[i].w,Agents[i].dw,l)
                else:
                    temp_price = random.randrange(10,100,1)
                temp_price = int(temp_price)
            else:
                temp_price = bid_option[Agents[i].action]
            if Agents[i].budget < temp_price:
                temp_price = Agents[i].budget
            bid_p.append(temp_price)
            Agents[i].bid_log.append(temp_price)
        total_market.append(np.sort(bid_p)[2])
        if len(bid_p) != 4:
            print("error")

        reward_1 = []
        for i in range(4):
            if bid_p[i] == np.max(bid_p):
                reward_1.append(5)
                if i == 2 or i == 3:
                    Agents[i].update_w_dw(bid_price=bid_p[i], flag=1,request=request)
                second_price = np.sort(bid_p)[2]
                Agents[i].budget -= second_price
                Agents[i].interval.append(np.max(bid_p) - second_price)
                Agents[i].consumption.append(second_price)
                Agents[i].win_log.append(second_price)
                Agents[i].win += 1
                Agents[i].win_rate.append(1)
                Agents[i].win_period += 1
            else:
                if i == 2 or i == 3:
                    Agents[i].update_w_dw(bid_price=bid_p[i],flag=0,request=request)
                reward_1.append(-1)
                Agents[i].consumption.append(0)
                Agents[i].win_rate.append(0)
            Agents[i].budget_log.append(Agents[i].budget)

        for i in range(4):
            win_rate = np.sum(Agents[i].win_rate) / (len(Agents[i].win_rate)+0.001)
            consumption_rate = np.mean(Agents[i].consumption)
            remaining_budget = Agents[i].budget
            Agents[i].next_state[3] = Agents[i].state[3] + 1
            Agents[i].next_state[2] = win_rate
            Agents[i].next_state[1] = consumption_rate
            Agents[i].next_state[0] = remaining_budget
            if len(Agents[i].state) == 5:
                Agents[i].next_state[4] = 0

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
                    Agents[i].reward = reward_1[i] - 500*Agents[i].get_lambda(Agents[i].w,Agents[i].dw,bid_p[3])
                    if request % 1000 == 0:
                        print(500*Agents[i].get_lambda(Agents[i].w,Agents[i].dw,bid_p[3]))
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

        if request % 2000 == 0:
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
              "bid_price3": Agents[3].bid_log,
              "A1_b_log":Agents[0].budget_log,"A2_b_log":Agents[1].budget_log,"A3_b_log":Agents[2].budget_log,"A4_b_log":Agents[3].budget_log}
    output = pd.DataFrame(output)
    output.to_excel(writer,sheet_name='{}'.format(time))

    total_request += request
    print(epsilon)
    for i in range(4):
        print(Agents[i].budget)
for i in range(4):
    torch.save(Agents[i].network.state_dict(), "./pth/MARL/agent{}.pt".format(i))
    # ttotal_win.append(Agents[i].win_log)
writer.save()

# print("train win:",len(ttotal_win[0]),"total request:",len(data),"win percent:",len(ttotal_win[0])/len(data))

# # win price distribution (our client)
# for i in range(len(Our_client.total_win)):
#     plt.hist(Our_client.total_win[i], bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
#     plt.hist(data,bins=40,facecolor="green",edgecolor="black",alpha=0.7)
# plt.show()
