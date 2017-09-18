import torch
from data.DataLoader import parse_data
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from data.mldata import loader


class Adaptive_Para_Pingpong(nn.Module):
    def __init__(self):
        super(Adaptive_Para_Pingpong, self).__init__()
        self.kd = Variable(torch.FloatTensor([0.18]), requires_grad=True)
        self.km = Variable(torch.FloatTensor([0.0003]), requires_grad=True)
        self.kv = Variable(torch.FloatTensor([0.99]), requires_grad=True)
        self.mu = Variable(torch.FloatTensor([0.22]), requires_grad=True)
        self.g = Variable(torch.FloatTensor([9.8015]), requires_grad=False)
        self.r = 0.02
        self.wok = True

    def update_pos(self, last_pos, last_v, delta_t):
        new_pos = last_pos + last_v * delta_t
        return new_pos

    def update_v(self, last_v, v_value, delta_t):
        mask = Variable(torch.FloatTensor([0, 0, 1]), requires_grad=False)
        maskd = Variable(torch.FloatTensor([1., 1., 1.]))
        new_v = last_v + ((- self.kd * v_value * maskd) * last_v - mask * self.g) * delta_t

        # print torch.min(torch.abs(self.wx), torch.abs(self.wy), torch.abs(self.wz))> 5000
        '''
        if self.wok:
            new_v[0] = new_v[0] - self.km * self.wz * last_v[1] * delta_t + self.km * self.wy * last_v[2] * delta_t
            new_v[1] = new_v[1] + self.km * self.wz * last_v[0] * delta_t - self.km * self.wx * last_v[2] * delta_t
            new_v[2] = new_v[2] - self.km * self.wz * last_v[0] * delta_t + self.km * self.wx * last_v[1] * delta_t
        '''
        return new_v

    def forward(self, input):
        t_seq, x_seq, y_seq, z_seq = input[:, 0], input[:, 1], input[:, 2], input[:, 3]
        bounce_id = 0
        bounce_loss = 0.

        lowest_z = 100.
        for i in range(z_seq.shape[0]):
            if lowest_z > z_seq[i]:
                lowest_z = z_seq[i]
                bounce_id = i

        bounce_point = Variable(torch.FloatTensor([x_seq[bounce_id], y_seq[bounce_id], z_seq[bounce_id]]))

        self.final_y = 3.0
        # print(self.final_y)
        poly_dim = 3
        sample_nums = 8
        '''
        sample_nums = max(sample_nums, x_seq.shape[0] / 3)
        cur_sam = 0
        while t_seq[cur_sam] < 0.24 and z_seq[cur_sam] > 0.05:
            cur_sam = cur_sam + 1
        sample_nums = max(cur_sam, sample_nums)
        '''
        p_x_fit = np.polyfit(t_seq[:sample_nums], x_seq[:sample_nums], poly_dim)
        tmp_t_seq = np.linspace(start=t_seq[0], stop=t_seq[sample_nums - 1], num=30)
        p_x_pre = np.polyval(p_x_fit, tmp_t_seq)
        v_x_fit = p_x_fit[:poly_dim] * np.array(range(poly_dim, 0, -1), dtype=float)
        v_x_pre = np.polyval(v_x_fit, tmp_t_seq)
        v_x_fit = np.polyfit(tmp_t_seq, v_x_pre, poly_dim)
        a_x_fit = v_x_fit[:poly_dim] * np.array(range(poly_dim, 0, -1), dtype=float)
        a_x_pre = np.polyval(a_x_fit, tmp_t_seq)
        a_x_fit = np.polyfit(tmp_t_seq, a_x_pre, poly_dim)
        a_ba_x_fit = a_x_fit[:poly_dim] * np.array(range(poly_dim, 0, -1), dtype=float)
        a_ba_x_pre = np.polyval(a_ba_x_fit, tmp_t_seq)

        p_y_fit = np.polyfit(t_seq[:sample_nums], y_seq[:sample_nums], poly_dim)
        p_y_pre = np.polyval(p_y_fit, tmp_t_seq)
        v_y_fit = p_y_fit[:poly_dim] * np.array(range(poly_dim, 0, -1), dtype=float)
        v_y_pre = np.polyval(v_y_fit, tmp_t_seq)
        v_y_fit = np.polyfit(tmp_t_seq, v_y_pre, poly_dim)
        a_y_fit = v_y_fit[:poly_dim] * np.array(range(poly_dim, 0, -1), dtype=float)
        a_y_pre = np.polyval(a_y_fit, tmp_t_seq)
        a_y_fit = np.polyfit(tmp_t_seq, a_y_pre, poly_dim)
        a_ba_y_fit = a_y_fit[:poly_dim] * np.array(range(poly_dim, 0, -1), dtype=float)
        a_ba_y_pre = np.polyval(a_ba_y_fit, tmp_t_seq)

        p_z_fit = np.polyfit(t_seq[:sample_nums], z_seq[:sample_nums], poly_dim)
        p_z_pre = np.polyval(p_z_fit, tmp_t_seq)
        v_z_fit = p_z_fit[:poly_dim] * np.array(range(poly_dim, 0, -1), dtype=float)
        v_z_pre = np.polyval(v_z_fit, tmp_t_seq)
        v_z_fit = np.polyfit(tmp_t_seq, v_z_pre, poly_dim)
        a_z_fit = v_z_fit[:poly_dim] * np.array(range(poly_dim, 0, -1))
        a_z_pre = np.polyval(a_z_fit, tmp_t_seq)
        a_z_fit = np.polyfit(tmp_t_seq, a_z_pre, poly_dim)
        a_ba_z_fit = a_z_fit[:poly_dim] * np.array(range(poly_dim, 0, -1), dtype=float)
        a_ba_z_pre = np.polyval(a_ba_z_fit, tmp_t_seq)

        vx, vy, vz = Variable(torch.FloatTensor(v_x_pre[15:16])), Variable(torch.FloatTensor(v_y_pre[15:16])), Variable(
            torch.FloatTensor(v_z_pre[15:16]))
        v_value = torch.sqrt(torch.sum(vx ** 2 + vy ** 2 + vz ** 2))
        ax, ay, az = Variable(torch.FloatTensor(a_x_pre[15:16])), Variable(torch.FloatTensor(a_y_pre[15:16])), Variable(
            torch.FloatTensor(a_z_pre[15:16]))
        a_ba_x, a_ba_y, a_ba_z = Variable(torch.FloatTensor(a_ba_x_pre[15:16])), Variable(
            torch.FloatTensor(a_ba_y_pre[15:16])), Variable(
            torch.FloatTensor(a_ba_z_pre[15:16]))

        vx, vy = np.polyfit(t_seq[:8], x_seq[:8], 1)[0], np.polyfit(t_seq[:8], y_seq[:8], 1)[0]
        pz_fit = np.polyfit(t_seq[:8], z_seq[:8], 2)
        vz = np.polyval(pz_fit.T[:2] * np.array(range(2, 0, -1), dtype=float), t_seq[4])
        vx, vy, vz = Variable(torch.FloatTensor([vx])), Variable(torch.FloatTensor([vy])), Variable(
            torch.FloatTensor([vz]))
        v_value = torch.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        ax, ay, az = Variable(torch.FloatTensor([0.])), -self.kd * v_value * vy, -self.g
        a_ba_x, a_ba_y, a_ba_z = Variable(torch.FloatTensor([0.])), Variable(torch.FloatTensor([0.])), Variable(
            torch.FloatTensor([0.]))
        a_ba_value = 0.
        # print('a')
        # print(ax,ay,az)
        # print('v')
        # print(vx,vy,vz)
        v_ba_value = (vx * ax + vy * ay + vz * az) / (torch.sqrt(vx ** 2 + vy ** 2 + vz ** 2))
        # print(v_ba_value)
        # print(type(v_ba_value))
        wy = (ay * (ax + self.kd * v_value * vx) - vy * (
        a_ba_x + self.kd * v_ba_value * vx + self.kd * v_value * ax)) / (self.km * (vz * ay - vy * az))
        wz = (vz * wy - (ax + self.kd * v_value * vx) / self.km) / vy
        wx = (vx * wy + (az + self.kd * v_value * vz + self.g) / self.km) / vy
        self.wx, self.wy, self.wz = (wx, wy, wz)
        self.wok = True
        # print(vx, vy,vz)
        # print(wx,wy,wz)
        # print torch.min(torch.abs(self.wx), torch.abs(self.wy), torch.abs(self.wz)).data[0] > 5000
        if torch.min(torch.abs(self.wx), torch.abs(self.wy), torch.abs(self.wz)).data[0] > 5000 or \
                        torch.max(torch.abs(self.wx), torch.abs(self.wy), torch.abs(self.wz)).data[0] > 10000:
            self.wx, self.wy, self.wz = Variable(torch.FloatTensor([0.])), Variable(torch.FloatTensor([0.])), Variable(
                torch.FloatTensor([0.]))
            self.wok = False
        # print p_x_pre[15]
        last_pos = Variable(torch.FloatTensor([p_x_pre[15], p_y_pre[15], p_z_pre[15]]))
        last_v = Variable(torch.FloatTensor([vx.data[0], vy.data[0], vz.data[0]]))
        # print(last_v)
        delta_t = 0.0001
        have_bounced = False
        new_t_seq, new_p_seq = [], []
        new_v_seq = []
        new_p_fig_seq = []
        new_v_seq.append(last_v)
        new_p_seq.append(last_pos)
        new_p_fig_seq.append(last_pos.data.tolist())
        cnt = 0
        T_Re = False
        while True:
            if cnt > 10000:
                T_Re = True
                break
            v_value = torch.sqrt(torch.sum(last_v ** 2))

            bounce_result = self.bounce_check(last_pos, last_v, have_bounced)
            if bounce_result is not None:
                last_v = bounce_result
                have_bounced = True
                new_v_seq.append(last_v)
                new_p_seq.append(last_pos)

                new_p_fig_seq.append(last_pos.data.tolist())
                # print('#########################')
                # print(bounce_point, last_pos)
                bounce_loss = torch.sqrt(torch.sum((bounce_point - last_pos) ** 2))

                cnt = cnt + 1
                continue

            # if have_bounced:
            #    print last_v
            if self.end_check(last_pos) is True:
                break
            if have_bounced and last_pos[2].data[0] < 0.01985:
                break
            last_pos = self.update_pos(last_pos, last_v, delta_t)
            last_v = self.update_v(last_v, v_value, delta_t)
            new_v_seq.append(last_v)
            new_p_seq.append(last_pos)

            new_p_fig_seq.append(last_pos.data.tolist())
            cnt = cnt + 1
        if T_Re:
            have_bounced = False
            cnt = 0
            last_pos = Variable(torch.FloatTensor([x_seq[0], y_seq[0], z_seq[0]]))
            last_v = Variable(torch.FloatTensor([vx.data[0], vy.data[0], vz.data[0]]))
            self.wx = Variable(torch.FloatTensor([50000.]))
            self.wy = Variable(torch.FloatTensor([50000.]))
            self.wz = Variable(torch.FloatTensor([50000.]))
            new_v_seq = [last_v]
            new_p_seq = [last_pos]
            self.wok = False

            while True:
                cnt = cnt + 1
                if cnt > 13000:
                    break
                v_value = torch.sqrt(torch.sum(last_v ** 2))

                bounce_result = self.bounce_check(last_pos, last_v, have_bounced)
                if bounce_result is not None:
                    last_v = bounce_result
                    have_bounced = True
                    new_v_seq.append(last_v)
                    new_p_seq.append(last_pos)
                    # print('#########################')
                    # print(bounce_point, last_pos)
                    bounce_loss = torch.sqrt(torch.sum((bounce_point - last_pos) ** 2))

                    cnt = cnt + 1
                    continue
                if self.end_check(last_pos) is True:
                    break
                if have_bounced and last_pos[2].data[0] < 0.01985:
                    break
                last_pos = self.update_pos(last_pos, last_v, delta_t)
                last_v = self.update_v(last_v, v_value, delta_t)
                new_v_seq.append(last_v)
                new_p_seq.append(last_pos)
                cnt = cnt + 1
        # print last_pos

        while have_bounced == False:
            last_pos[1] = 2.7
            while True:
                if cnt > 10000:
                    break
                v_value = torch.sqrt(torch.sum(last_v ** 2))
                self.wok = False
                bounce_result = self.bounce_check(last_pos, last_v, have_bounced)
                if bounce_result is not None:
                    last_v = bounce_result
                    have_bounced = True
                    new_v_seq.append(last_v)
                    new_p_seq.append(last_pos)

                    new_p_fig_seq.append(last_pos.data.tolist())
                    # print('#########################')
                    # print(bounce_point, last_pos)
                    bounce_loss = torch.sqrt(torch.sum((bounce_point - last_pos) ** 2))

                    cnt = cnt + 1
                    continue

                # if have_bounced:
                #    print last_v
                if self.end_check(last_pos) is True:
                    break
                if have_bounced and last_pos[2].data[0] < 0.01985:
                    break
                last_pos = self.update_pos(last_pos, last_v, delta_t)
                last_v = self.update_v(last_v, v_value, delta_t)
                new_v_seq.append(last_v)
                new_p_seq.append(last_pos)

                new_p_fig_seq.append(last_pos.data.tolist())
                cnt = cnt + 1

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(x_seq, y_seq, z_seq)
        # print (new_p_seq[10])
        new_p_fig_seq = np.array(new_p_fig_seq)
        ax.scatter(new_p_fig_seq[:, 0], new_p_fig_seq[:, 1], new_p_fig_seq[:, 2])
        plt.show()

        hit_pos = last_pos
        gt_pos = Variable(torch.FloatTensor([x_seq[-1], y_seq[-1], z_seq[-1]]))
        hit_loss = torch.sqrt(torch.sum((hit_pos - gt_pos) ** 2))

        '''
        if bounce_loss.data[0] < 0.00001 :
            print('###########################################################')
        '''
        gt_time = t_seq[-1]
        pred_time = cnt * delta_t
        time_loss = pred_time - gt_time
        print(cnt, bounce_loss, hit_loss, time_loss)
        return hit_loss, bounce_loss, abs(time_loss)

    def bounce_check(self, last_pos, last_v, have_bounced):
        px, py, pz = last_pos[0], last_pos[1], last_pos[2]
        # mask = torch.lt(last_pos,0.0205)
        if have_bounced is False and last_pos[2].data[0] < 0.0205:
            vx, vy, vz = last_v[0], last_v[1], last_v[2]
            # if self.wok is False:
            vex = 0.648819 * vx + 0.010491
            vey = 0.643550 * vy + 0.024064
            vez = -0.807392 * vz + 0.468110
            new_v = Variable(torch.FloatTensor(3))
            # print 'not ok'
            new_v[0], new_v[1], new_v[2] = vex, vey, vez
            return new_v
            '''
            bounce_condition = self.mu * (1 + self.kv) * torch.abs(vz) / (torch.sqrt((vx - self.wy * self.r) ** 2  + (vy + self.wx * self.r) ** 2))
            if bounce_condition.data[0] <= 0.4:
                vex = bounce_condition * (self.wy * self.r - vx) + vx
                vey = bounce_condition * (-self.wx * self.r - vy) + vy
                vez = -self.kv * vz
                wex = self.wx + (3 * bounce_condition * (-self.wx * self.r - vy) / (2 * self.r))
                wey = self.wy - (3 * bounce_condition * (self.wy * self.r - vx) / (2 * self.r))
                wez = self.wz
                self.wx, self.wy, self.wz = wex, wey, wez
                new_v = Variable(torch.FloatTensor(3))
                new_v[0], new_v[1], new_v[2] = vex, vey, vez
                return new_v
            else:
                vex = 0.4 * self.wy * self.r + 0.6 * vx
                vey = -0.4 * self.wx * self.r + 0.6 * vy
                vez = -self.kv * vz
                self.wx = 0.4 * self.wx - 0.6 * vy / self.r
                self.wy = 0.4 * self.wy + 0.6 * vx / self.r
                self.wz = self.wz
                new_v = Variable(torch.FloatTensor(3))
                new_v[0], new_v[1], new_v[2] = vex, vey, vez
                return new_v
            '''
        return None

    def end_check(self, last_pos):
        if last_pos[1].data[0] > self.final_y:
            return True
        else:
            return False


if __name__ == "__main__":
    '''
    dflist, ydf = loader()
    #print(ydf.iloc[0,:])
    for i in range(len(dflist)):
        dflist[i] = np.array(dflist[i].tolist(),dtype=float)
        dflist[i] = np.vstack((dflist[i],np.array(ydf.iloc[i,0:4], dtype=float)))
        dflist[i] = np.vstack((dflist[i],np.array(ydf.iloc[i,4:8], dtype=float)))
    train_data = dflist
    #print(type(train_data[0]))
    table_length = 2.74
    table_width = 1.525
    for i in range(len(train_data)):
        train_data[i][:, 1] = train_data[i][:, 1] + table_width / 2
        train_data[i][:, 2] = 3 - (train_data[i][:, 2] + table_length / 2)
    '''
    train_data = parse_data()
    for batch_ in range(len(train_data)):
        train_data[batch_][:, 1:] /= 1000
        train_data[batch_][:, 2] = 3 - train_data[batch_][:, 2]

    model = Adaptive_Para_Pingpong()
    optimizers = optim.SGD(params=[model.kd, model.km, model.kv, model.mu], lr=0.001, weight_decay=0.0005, momentum=0.9)
    for epoch in range(1):
        print('epoch id is ', epoch)
        total_loss = 0.
        total_time_loss = 0.
        total_bounce_loss = 0.
        total_number = 10
        for batch_ in range(total_number):
            # if batch_ == 50:
            #    total_number = total_number - 1
            #    continue
            # if batch_ % 10 == 0:
            # optimizers.zero_grad()
            print('batch id is ', batch_)
            train_batch = train_data[batch_][:, :]
            batch_loss, bounce_loss, time_loss = model(train_batch)
            total_loss += batch_loss.data[0]
            total_bounce_loss += bounce_loss
            total_time_loss += time_loss
            # batch_loss.backward()
            # optimizers.step()
        print('total hit loss is ', total_loss / total_number)
        print('total bounce loss is ', total_bounce_loss / total_number)
        print('total time loss ', total_time_loss / total_number)
        print(model.kd, model.km, model.kv, model.mu)
