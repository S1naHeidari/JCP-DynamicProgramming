import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt


class problem_parameters:
    @staticmethod
    def max_cars():
        return 20

    def movement_cost():
        return -2

    def credit():
        return 10
    
    def discount_factor():
        return 0.9

values = np.zeros((problem_parameters.max_cars()+1,problem_parameters.max_cars()+1))
policy = values.copy().astype(int)

class Poisson:
    
    def __init__(self, λ):
        self.λ = λ
        self.vals = {}
        self.ɑ = 0
        minimum = 0.01
        summer = 0
        while 1:
            p = poisson.pmf(k=self.ɑ ,mu=λ)
            if p >= 0.01:
                self.vals[self.ɑ] = p
                summer += p
                break
            self.ɑ+=1

        
        self.ꞵ = self.ɑ + 1
        while 1: 
            p = poisson.pmf(k=self.ꞵ ,mu=λ)
            if  p >= 0.01:
                self.ꞵ += 1
                self.vals[self.ꞵ] = p    
                summer += p

            else:
                break

        added_val = (1-summer)/(self.ꞵ - self.ɑ)
        for key in self.vals:
            self.vals[key] += added_val
            
    def values(self):
        return self.vals


class Location:
    def __init__(self, rent_mu, return_mu):
        self.rent_mu = rent_mu
        self.return_mu = return_mu
        self.poisson_rent = Poisson(self.rent_mu).values()
        self.poisson_return = Poisson(self.return_mu).values()
    
location1 = Location(3, 4)
location2 = Location(3, 2)
pass


def make_action(state, action):
    if action < 0:
        # moving from loc 2 to loc 1
        loc1 = min(state[0] + min(abs(action), state[1]), problem_parameters.max_cars())
        loc2 = state[1] - min(abs(action), state[1])
    if action >= 0:
        loc1 = state[0] - min(abs(action), state[0])
        loc2 = min(state[1] + min(abs(action), state[0]), problem_parameters.max_cars())

    return loc1, loc2

m = make_action((18, 18), -5)

def expected_reward(state_tuple, policy):
    reward = 0

    # reward for moving cars based on current policy (-2 for each car)
    new_state = make_action(state_tuple, policy)
    if policy >= 0:
        movement = abs(state_tuple[0]-new_state[0])
    else:
        movement = abs(state_tuple[1]-new_state[1])
    
    reward = movement * problem_parameters.movement_cost()
    pass
    for rent in location1.poisson_rent:
        for rent2 in location2.poisson_rent:
            for ret in location1.poisson_return:
                for ret2 in location2.poisson_return:
                    if rent == 6 and rent2 == 5 and ret == 4 and ret2 == 6:
                        pass
                    new_s = [new_state[0], new_state[1]]
                    # for rents
                    new_s[0] = max(new_s[0] - rent, 0)
                    new_s[1] = max(new_s[1] - rent2, 0)
                    credit = abs(new_state[0]-new_s[0])
                    credit += abs(new_state[1]-new_s[1])
                    credit *= problem_parameters.credit()
                    # for returns
                    new_s[0] = min(new_s[0] + ret, problem_parameters.max_cars())
                    new_s[1] = min(new_s[1] + ret2, problem_parameters.max_cars())
                    
                    prob = location1.poisson_rent[rent] * location1.poisson_return[ret] * location2.poisson_rent[rent2] * location2.poisson_return[ret2]
                    reward += prob * (credit + problem_parameters.discount_factor()*values[new_s[0]][new_s[1]])
                    pass

    #print(reward)
    return reward


expected_reward((18, 15), -3)

def policy_evaluation():
    global values, policy
    ε = policy_evaluation.ε
    policy_evaluation.ε /= 10
    while(1):
        Δ = 0
        # find the maximum value of Δ in this nested loop 
        for i in range(0, problem_parameters.max_cars()+1):
            for j in range(0, problem_parameters.max_cars()+1):
                recent = values[i][j]
                values[i][j] = expected_reward((i,j), policy[i][j])
                Δ = max(Δ , abs(values[i][j] - recent))
                print('.', end = '')
        print(Δ)
        if Δ < ε:
            break
            
    
policy_evaluation.ε = 50

def policy_improvement():
    global values, policy
    policy_stable = True
    for i in range(0, problem_parameters.max_cars()+1):
        for j in range(0, problem_parameters.max_cars()+1):
            old_action = policy[i][j]
            max_action = policy[i][j]
            max_reward = expected_reward((i,j), max_action)
            for action in range(-5,6):
                if action < 0:
                    new_state = (i+min(abs(action), j), j-min(abs(action), j))
                else:
                    new_state = (i-min(abs(action), i), j+min(abs(action), i))
                σ = expected_reward((new_state[0],new_state[1]), action)
                if σ > max_reward:
                    max_reward = σ
                    policy[i][j] = action
                    max_action = action
            if max_action != old_action:
                policy_stable = False
    return policy_stable


def save_policy():
    save_policy.counter += 1
    ax = sns.heatmap(policy, linewidth=0.5)
    ax.invert_yaxis()
    plt.savefig('policy'+str(save_policy.counter)+'.svg')
    plt.close()
    
def save_value():
    save_value.counter += 1
    ax = sns.heatmap(values, linewidth=0.5)
    ax.invert_yaxis()
    plt.savefig('value'+ str(save_value.counter)+'.svg')
    plt.close()


# In[41]:


save_policy.counter = 0
save_value.counter = 0

def main():
    while(1):
        policy_evaluation()
        is_stable = policy_improvement()
        save_value()
        save_policy()
        if is_stable:
            break

if __name__ == '__main__':
    main()
