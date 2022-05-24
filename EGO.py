import time
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from smt.surrogate_models import KRG
from smt.sampling_methods import LHS
from tqdm import trange


def branin_func(x):
    """
    branin函数式
    :param x: 由x1,x2组成的数组，形式为[x1, x2]
    :return: 由x1,x2对应的函数值
    """
    arg0 = (x[1] - (5 / (4 * np.pi ** 2)) * x[0] ** 2 + (5 / np.pi) * x[0] - 6) ** 2
    arg1 = 10 * (1 - (1 / (8 * np.pi))) * np.cos(x[0]) + 10
    return np.array([arg0 + arg1])


def con(x_limits):
    """
    Branin函数俩变量的约束条件
    :param x_limits: 变量上下界
    :return: 约束条件
    """
    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - x_limits[0][0]},
            {'type': 'ineq', 'fun': lambda x: -x[0] + x_limits[0][1]},
            {'type': 'ineq', 'fun': lambda x: x[1] - x_limits[1][0]},
            {'type': 'ineq', 'fun': lambda x: -x[1] + x_limits[1][1]})
    return cons


class EGO:
    def __init__(self, model, n_iter, x1_data, x2_data, sample_data, cons):
        self.model = model
        self.n_iter = n_iter
        self.x1_data = x1_data
        self.x2_data = x2_data
        self.sample_data = sample_data
        self.cons = cons
        self.x_data = self.get_x_data()
        self.y_data = self.get_y_data()
        self.opt_x, self.opt_y, self.pred_res = [], [], []
        self.y_min = None
        self.iter_num_end = None
        self.error = None
        self.global_min = 0.397887

    def Ei(self, points, f_min):
        """
        预期改进函数
        :param points: 二维数组，shape为[num_data, ndim]
        :param f_min: 当前求得的最小函数值
        :return: 预期改进函数值
        """
        if points.ndim == 1:
            points = np.expand_dims(points, 0)

        # 预测值与方差
        pred = self.model.predict_values(points)
        var = self.model.predict_variances(points)

        args0 = (f_min - pred) / np.sqrt(var)
        args1 = (f_min - pred) * norm.cdf(args0)
        args2 = np.sqrt(var) * norm.pdf(args0)

        # 当输入数据仅有一个时
        if var.size == 1 and var == 0.0:
            return np.array([[0.0]])
        ei = args1 + args2
        return ei

    def ego_optim(self):
        time_start = time.time()
        x_sample_data, y_sample_data = self.sample_data[0], self.sample_data[1]
        # 0, 5:11, 6
        np.random.seed(6)
        for k in trange(self.n_iter):
            # 求解预期改进函数的初始点初始化
            x1_init = np.expand_dims(np.random.rand(1000) * 15 - 5, 1)
            x2_init = np.expand_dims(np.random.rand(1000) * 15, 1)
            x_init = np.concatenate((x1_init, x2_init), axis=1)

            # 找到当前采样数据的最小值y
            y_min_k = np.min(y_sample_data)

            # 对当前采样数据进行训练
            self.model.set_training_values(x_sample_data, y_sample_data)
            self.model.train()

            # 将目标函数转为-Ei, 便为求其最小值
            obj_k = lambda x: -self.Ei(x, y_min_k)[:, 0]

            # 采用非线性规划求解在不同初始点下的目标函数最小值，
            opt_all = np.array([minimize(obj_k, x_i, method='SLSQP', constraints=self.cons) for x_i in x_init])
            # 过滤掉未成功的优化
            opt_succ = opt_all[[opt_i["success"] for opt_i in opt_all]]
            # 获取成功优化后的不同初始点下的目标函数最小值
            opt_succ_values = [opt_i["fun"] for opt_i in opt_succ]
            opt_values = np.array([value[0] if type(value) is np.ndarray else value for value in opt_succ_values])

            # 寻找所有初始点下，所求较小值里的最小值索引
            obj_min_id = np.argmin(opt_values)

            # 获取最小值对应的x值
            x_k = opt_succ[obj_min_id]["x"]
            y_k = branin_func(x_k)

            self.opt_x.append([x_k[0], x_k[1]])
            self.opt_y.append(y_k[0])

            # 更新采样数据
            x_sample_data = np.concatenate((x_sample_data, np.expand_dims(x_k, 0)), axis=0)
            y_sample_data = np.concatenate((y_sample_data, np.expand_dims(y_k, 0)), axis=0)

            # 计算第k轮迭代下的模型预测值
            y_pred_k = self.model.predict_values(self.x_data).reshape((150, 150)).T
            self.pred_res.append(y_pred_k)

            # 计算最小值与全局最小值的误差
            self.error = abs(min(y_min_k, y_k[0]) - self.global_min) / self.global_min
            # 停止条件
            if (abs(y_k - y_min_k) / y_min_k) < 0.01 and self.error < 0.01:
                self.y_min = min(y_min_k, y_k[0])
                self.iter_num_end = k + 1
                break

        time_end = time.time()
        print("Algorithm optimization ends.")
        print("The time taken by the algorithm is:{:.2f}s".format(time_end - time_start))
        print("The actual number of iterations of the algorithm is: {}".format(self.iter_num_end))
        print("At this time, the minimum value of the Branin function is: {:.6f}".format(self.y_min))
        print("The error between the obtained minimum value and the true value is: {:.2f}%".format(self.error * 100))

    def get_x_data(self):
        x_data = []
        for x1_i in self.x1_data:
            for x2_i in self.x2_data:
                x_data.append(np.array([x1_i, x2_i]))
        return np.array(x_data)

    def get_y_data(self):
        y_data = []
        for x1_i in self.x1_data:
            for x2_i in self.x2_data:
                y_data.append(branin_func(np.array([x1_i, x2_i])))
        return np.array(y_data).reshape((150, 150)).T

    def plot(self):
        x_mesh, y_mesh = np.meshgrid(self.x1_data, self.x2_data)

        fig1 = plt.figure(figsize=[6, 3 * (self.iter_num_end // 2)])
        fig2 = plt.figure(figsize=[6, 3 * (self.iter_num_end // 2)])
        fig3 = plt.figure(figsize=[12, 9])

        for i, (x_i, pred_i) in enumerate(zip(self.opt_x, self.pred_res)):
            ax1 = fig1.add_subplot((self.iter_num_end + 1) // 2, 2, i + 1)
            ax2 = fig2.add_subplot((self.iter_num_end + 1) // 2, 2, i + 1, projection='3d')

            ax1.contour(x_mesh, y_mesh, self.y_data, 40)
            ax1.scatter(self.sample_data[0][:20 + i, 0], self.sample_data[0][:20 + i, 1], color='w', edgecolors='b',
                        linewidths=2)
            ax1.scatter(x_i[0], x_i[1], marker="*", color="red")
            ax1.set_title("iteration {}".format(i + 1))

            ax2.plot_surface(y_mesh, x_mesh, self.y_data, linewidth=0, antialiased=False, alpha=0.5)
            ax2.plot_surface(y_mesh, x_mesh, pred_i, linewidth=0, antialiased=False, alpha=0.5)
            ax2.set_title("iteration {}".format(i + 1))

        ax3 = fig3.add_subplot(111)
        ax3.plot(self.opt_y, marker="o")
        ax3.set_xlabel('Number of Iterations')
        ax3.set_ylabel('$y_i$')

        fig1.tight_layout()
        fig2.tight_layout()
        fig3.tight_layout()

        fig1.savefig("./figure/Function Contour Plot and Point Selection.jpg", dpi=800, bbox_inches="tight")
        fig2.savefig("./figure/True and Predicted Surfaces.jpg", dpi=800, bbox_inches="tight")
        fig3.savefig("./figure/Function value of selected point.jpg", dpi=800, bbox_inches="tight")
        print("Image drawing ends.")


if __name__ == "__main__":
    problem_dim = 2
    # Kriging代理模型
    model = KRG(theta0=[1e-2] * problem_dim, print_prediction=False)

    num_data = 150
    xlimits = np.array([[-5, 10], [0, 15]])
    x1 = np.linspace(xlimits[0][0], xlimits[0][1], num_data)
    x2 = np.linspace(xlimits[1][0], xlimits[1][1], num_data)

    # LHS采样20个数据，
    sampling = LHS(xlimits=xlimits, random_state=1)
    sample_x = sampling(20)
    sample_y = np.array([branin_func(x) for x in sample_x])
    sample_data = tuple((sample_x, sample_y))

    cons = con(xlimits)

    ego = EGO(model, 100, x1, x2, sample_data, cons)
    ego.ego_optim()
    ego.plot()
