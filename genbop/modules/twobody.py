import torch
import numpy as np


class MorseEnergy(torch.nn.Module):
    def __init__(self,ntype_edge):
        super().__init__()

        self.re = torch.nn.Parameter(torch.empty(ntype_edge))
        self.De = torch.nn.Parameter(torch.empty(ntype_edge))
        self.a = torch.nn.Parameter(torch.empty(ntype_edge))
        self.positive_constraint = torch.nn.Softplus(beta=np.log(2))
        self.inverse_softplus = InverseSoftplus(beta=np.log(2))

    def set_default_parameters(self,zs,index_map,elem_node2edge):

        indices = index_map[zs]
        count = 0
        for z1 in zs:
            i1 = index_map[z1]
            for z2 in zs:
                i2 = index_map[z2]
                torch.nn.init.constant_(self.De[elem_node2edge[i1,i2]], self.inverse_softplus.forward(np.clip(-morse_dict[tuple(sorted([z1,z2]))][0], 0.01, None)))
                torch.nn.init.constant_(self.re[elem_node2edge[i1,i2]], self.inverse_softplus.forward(np.clip(morse_dict[tuple(sorted([z1,z2]))][1], 0.01, None)))
                torch.nn.init.constant_(self.a[elem_node2edge[i1,i2]], self.inverse_softplus.forward(np.clip(morse_dict[tuple(sorted([z1,z2]))][2], 0.01, None)))
    
    def set_random_parameters(self):

        torch.nn.init.uniform_(self.De)
        torch.nn.init.uniform_(self.re)
        torch.nn.init.uniform_(self.a)
            
    def forward(self,dist,edge_type):

        re = self.positive_constraint(self.re)
        De = self.positive_constraint(self.De)
        a = self.positive_constraint(self.a)

        t = torch.exp(-a[edge_type]*(dist-re[edge_type]))

        return De[edge_type]*t.pow(2), 2*De[edge_type]*t

class AsymmetricMorseEnergy(torch.nn.Module):
    def __init__(self,ntype_edge):
        super().__init__()

        self.re = torch.nn.Parameter(torch.empty(ntype_edge))
        self.De = torch.nn.Parameter(torch.empty(ntype_edge))
        self.a1 = torch.nn.Parameter(torch.empty(ntype_edge))
        self.a2 = torch.nn.Parameter(torch.empty(ntype_edge))
        self.positive_constraint = torch.nn.Softplus(beta=np.log(2))
        self.inverse_softplus = InverseSoftplus(beta=np.log(2))

    def set_default_parameters(self,zs,index_map,elem_node2edge):

        indices = index_map[zs]
        count = 0
        for z1 in zs:
            i1 = index_map[z1]
            for z2 in zs:
                i2 = index_map[z2]
                torch.nn.init.constant_(self.De[elem_node2edge[i1,i2]], self.inverse_softplus.forward(np.clip(-morse_dict[tuple(sorted([z1,z2]))][0], 0.01, None)))
                torch.nn.init.constant_(self.re[elem_node2edge[i1,i2]], self.inverse_softplus.forward(np.clip(morse_dict[tuple(sorted([z1,z2]))][1], 0.01, None)))
                torch.nn.init.constant_(self.a1[elem_node2edge[i1,i2]], self.inverse_softplus.forward(np.clip(morse_dict[tuple(sorted([z1,z2]))][2], 0.01, None)))
                torch.nn.init.constant_(self.a2[elem_node2edge[i1,i2]], self.inverse_softplus.forward(np.clip(morse_dict[tuple(sorted([z1,z2]))][2], 0.01, None)))
    
    def set_random_parameters(self):

        torch.nn.init.uniform_(self.De)
        torch.nn.init.uniform_(self.re)
        torch.nn.init.uniform_(self.a1)
        torch.nn.init.uniform_(self.a2)
            
    def forward(self,dist,edge_type):

        re = self.positive_constraint(self.re)
        De = self.positive_constraint(self.De)
        a1 = self.positive_constraint(self.a1)
        a2 = self.positive_constraint(self.a2)

        t1 = torch.exp(-a1[edge_type]*(dist-re[edge_type]))
        t2 = torch.exp(-a2[edge_type]*(dist-re[edge_type]))

        return De[edge_type]*t1.pow(2), 2*De[edge_type]*t2

class CoulumbAsymmetricMorseEnergy(torch.nn.Module):
    def __init__(self,ntype_edge):
        super().__init__()

        self.re = torch.nn.Parameter(torch.empty(ntype_edge))
        self.De = torch.nn.Parameter(torch.empty(ntype_edge))
        self.a1 = torch.nn.Parameter(torch.empty(ntype_edge))
        self.a2 = torch.nn.Parameter(torch.empty(ntype_edge))
        self.Q = torch.nn.Parameter(torch.empty(ntype_edge))
        self.positive_constraint = torch.nn.Softplus(beta=np.log(2))
        self.inverse_softplus = InverseSoftplus(beta=np.log(2))

    def set_default_parameters(self,zs,index_map,elem_node2edge):

        indices = index_map[zs]
        count = 0
        for z1 in zs:
            i1 = index_map[z1]
            for z2 in zs:
                i2 = index_map[z2]
                torch.nn.init.constant_(self.De[elem_node2edge[i1,i2]], self.inverse_softplus.forward(np.clip(-morse_dict[tuple(sorted([z1,z2]))][0], 0.01, None)))
                torch.nn.init.constant_(self.re[elem_node2edge[i1,i2]], self.inverse_softplus.forward(np.clip(morse_dict[tuple(sorted([z1,z2]))][1], 0.01, None)))
                torch.nn.init.constant_(self.a1[elem_node2edge[i1,i2]], self.inverse_softplus.forward(np.clip(morse_dict[tuple(sorted([z1,z2]))][2], 0.01, None)))
                torch.nn.init.constant_(self.a2[elem_node2edge[i1,i2]], self.inverse_softplus.forward(np.clip(morse_dict[tuple(sorted([z1,z2]))][2], 0.01, None)))
        
        torch.nn.init.uniform_(self.Q)
    
    def set_random_parameters(self):

        torch.nn.init.uniform_(self.De)
        torch.nn.init.uniform_(self.re)
        torch.nn.init.uniform_(self.a1)
        torch.nn.init.uniform_(self.a2)
        torch.nn.init.uniform_(self.Q)
            
    def forward(self,dist,edge_type):

        re = self.positive_constraint(self.re)
        De = self.positive_constraint(self.De)
        a1 = self.positive_constraint(self.a1)
        a2 = self.positive_constraint(self.a2)
        Q = self.positive_constraint(self.Q)

        t1 = torch.exp(-a1[edge_type]*(dist-re[edge_type]))
        t2 = torch.exp(-a2[edge_type]*(dist-re[edge_type]))

        return (1+Q[edge_type]/dist)*De[edge_type]*t1.pow(2), 2*De[edge_type]*t2

class RepulsiveEnergy(torch.nn.Module):
    def __init__(self,ntype_edge):
        super().__init__()

        self.re = torch.nn.Parameter(torch.empty(ntype_edge))
        self.a = torch.nn.Parameter(torch.empty(ntype_edge))
        self.Q = torch.nn.Parameter(torch.empty(ntype_edge))
        self.positive_constraint = torch.nn.Softplus(beta=np.log(2))
        self.inverse_softplus = InverseSoftplus(beta=np.log(2))

    def set_default_parameters(self,zs,index_map,elem_node2edge):

        indices = index_map[zs]
        count = 0
        for z1 in zs:
            i1 = index_map[z1]
            for z2 in zs:
                i2 = index_map[z2]
                torch.nn.init.constant_(self.re[elem_node2edge[i1,i2]], self.inverse_softplus.forward(np.clip(morse_dict[tuple(sorted([z1,z2]))][1], 0.01, None)))
                torch.nn.init.constant_(self.a[elem_node2edge[i1,i2]], self.inverse_softplus.forward(np.clip(morse_dict[tuple(sorted([z1,z2]))][2], 0.01, None)))
        
        torch.nn.init.uniform_(self.Q)
    
    def set_random_parameters(self):

        torch.nn.init.uniform_(self.re)
        torch.nn.init.uniform_(self.a)
        torch.nn.init.uniform_(self.Q)
            
    def forward(self,dist,edge_type):

        re = self.positive_constraint(self.re)
        a = self.positive_constraint(self.a)
        Q = self.positive_constraint(self.Q)

        t1 = torch.exp(-a[edge_type]*(dist-re[edge_type]))

        return (1+Q[edge_type]/dist)*t1.pow(2)


class ExtendedCoulumbAsymmetricMorseEnergy(torch.nn.Module):
    def __init__(self,ntype_edge,nfunc=3):
        super().__init__()

        self.re = torch.nn.Parameter(torch.empty(ntype_edge,nfunc))
        self.De = torch.nn.Parameter(torch.empty(ntype_edge,nfunc))
        self.a1 = torch.nn.Parameter(torch.empty(ntype_edge,nfunc))
        self.a2 = torch.nn.Parameter(torch.empty(ntype_edge,nfunc))
        self.Q = torch.nn.Parameter(torch.empty(ntype_edge))
        self.positive_constraint = torch.nn.Softplus(beta=np.log(2))
        self.inverse_softplus = InverseSoftplus(beta=np.log(2))
    
    def set_random_parameters(self):

        torch.nn.init.uniform_(self.De)
        torch.nn.init.uniform_(self.re)
        torch.nn.init.uniform_(self.a1)
        torch.nn.init.uniform_(self.a2)
        torch.nn.init.uniform_(self.Q)
            
    def forward(self,dist,edge_type):

        re = self.positive_constraint(self.re)
        De = self.positive_constraint(self.De)
        a1 = self.positive_constraint(self.a1)
        a2 = self.positive_constraint(self.a2)
        Q = self.positive_constraint(self.Q)

        t1 = torch.exp(-a1[edge_type]*(dist.view(-1,1)-re[edge_type]))
        t2 = torch.exp(-a2[edge_type]*(dist.view(-1,1)-re[edge_type]))

        return (1+Q[edge_type]/dist)*(De[edge_type]*t1.pow(2)).sum(dim=1), (2*De[edge_type]*t2).sum(dim=1)



class InverseSoftplus():
    def __init__(self, beta: float = 1.0, threshold: float = 20.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, inputs):

        return (1/self.beta)*np.log(-1+np.exp(self.beta*inputs))

morse_dict = {
(1, 1) : [-4.52776107, 0.750809384064957, 1.0] ,
(1, 2) : [0.002005490000000021, 2.8359041077405984, 1.0] ,
(1, 3) : [-2.32198043, 1.6131281991211983, 1.0] ,
(1, 4) : [-2.44462813, 1.3545676545673164, 1.0] ,
(1, 5) : [-3.67657515, 1.2538835411233373, 1.0] ,
(1, 6) : [-3.6917075099999996, 1.1400012005256837, 1.0] ,
(1, 7) : [-3.86121803, 1.0516146478154438, 1.0] ,
(1, 8) : [-4.885204079999999, 0.9893474212833427, 1.0] ,
(1, 9) : [-6.34812912, 0.9358616923456157, 1.0] ,
(1, 10) : [0.0005180600000000372, 2.92597074973418, 1.0] ,
(1, 11) : [-1.8509973099999997, 1.903125465832455, 1.0] ,
(1, 12) : [-1.39730269, 1.754723352639954, 1.0] ,
(1, 13) : [-3.0965433599999996, 1.677473886622382, 1.0] ,
(1, 14) : [-3.10357897, 1.5430840644631125, 1.0] ,
(1, 15) : [-3.2634407800000003, 1.437446285709487, 1.0] ,
(1, 16) : [-3.8972155600000007, 1.3538401932281372, 1.0] ,
(1, 17) : [-4.719645689999999, 1.2850777761676526, 1.0] ,
(1, 18) : [0.017664070000000063, 2.8806256595920274, 1.0] ,
(1, 19) : [-1.78847937, 2.2271921719285923, 1.0] ,
(1, 20) : [-1.97734897, 1.9891391089363255, 1.0] ,
(2, 2) : [0.00017188000000000273, 6.426410790814729, 1.0] ,
(2, 3) : [-0.0007560500000000636, 4.765824399566144, 1.0] ,
(2, 4) : [0.0002938399999999966, 3.516548113590941, 1.0] ,
(2, 5) : [-0.052547780000000044, 2.679482599309053, 1.0] ,
(2, 6) : [-0.12461036000000014, 2.7135866797100845, 1.0] ,
(2, 7) : [-0.002579629999999611, 3.1070047001412795, 1.0] ,
(2, 8) : [1.83138181, 1.1923957374546421, 1.0] ,
(2, 9) : [-0.027758350000000043, 2.8641711769201224, 1.0] ,
(2, 10) : [0.0001086100000000003, 4.820574525593397, 1.0] ,
(2, 11) : [-0.001034170000000001, 5.111576381773044, 1.0] ,
(2, 12) : [0.00154191, 3.7820195408670227, 1.0] ,
(2, 13) : [-0.04385527, 3.329226818752366, 1.0] ,
(2, 14) : [-0.04696613999999999, 2.6068923499638417, 1.0] ,
(2, 15) : [-0.0003708099999999437, 3.540034722541575, 1.0] ,
(2, 16) : [1.38392333, 1.6910531649537222, 1.0] ,
(2, 17) : [-0.10475228999999997, 2.6580917718355774, 1.0] ,
(2, 18) : [-0.00510702, 3.7800276824383174, 1.0] ,
(2, 19) : [-0.00029551999999999357, 5.188773886266389, 1.0] ,
(2, 20) : [-0.0003024000000000013, 4.449794409217127, 1.0] ,
(3, 3) : [-0.8487040800000001, 2.761027551329396, 1.0] ,
(3, 4) : [-0.5914534800000001, 2.6137859121779656, 1.0] ,
(3, 5) : [-1.56948161, 2.1600059211029956, 1.0] ,
(3, 6) : [-2.9544925300000004, 1.9035411580262716, 1.0] ,
(3, 7) : [-2.01405662, 1.9001116974272854, 1.0] ,
(3, 8) : [-4.00451041, 1.7132580563067552, 1.0] ,
(3, 9) : [-6.12515811, 1.7819511913349368, 1.0] ,
(3, 10) : [-0.004043750000000018, 4.783110266625682, 1.0] ,
(3, 11) : [-0.7917076700000002, 2.908217228956599, 1.0] ,
(3, 12) : [-0.30434424000000004, 3.080469681769324, 1.0] ,
(3, 13) : [-0.9620890200000001, 2.6410657123971757, 1.0] ,
(3, 14) : [-1.91944164, 2.381587180915282, 1.0] ,
(3, 15) : [-1.88021664, 2.3473791774657964, 1.0] ,
(3, 16) : [-3.3099556599999995, 2.1665184321394544, 1.0] ,
(3, 17) : [-4.89219168, 2.0448418629077407, 1.0] ,
(3, 18) : [-0.0072033500000000215, 4.471912698029781, 1.0] ,
(3, 19) : [-0.05376123000000002, 4.07896233131663, 1.0] ,
(3, 20) : [-0.41726701, 3.4307423165839785, 1.0] ,
(4, 4) : [0.08660038, 2.0095773084656385, 1.0] ,
(4, 5) : [-1.53763471, 1.7092570189412708, 1.0] ,
(4, 6) : [-2.7472367799999997, 1.6765385791862948, 1.0] ,
(4, 7) : [-2.3636887399999997, 1.6108245715471314, 1.0] ,
(4, 8) : [-5.60333604, 1.3367968132816594, 1.0] ,
(4, 9) : [-6.68173995, 1.3775173277676038, 1.0] ,
(4, 10) : [-0.007050780000000005, 3.960386133030465, 1.0] ,
(4, 11) : [-0.44662151000000005, 2.8976690395385045, 1.0] ,
(4, 12) : [-0.24592455000000002, 2.910105164336849, 1.0] ,
(4, 13) : [-0.98629071, 2.4199694268110084, 1.0] ,
(4, 14) : [-1.87984901, 2.1071264099479174, 1.0] ,
(4, 15) : [-1.7605522299999996, 2.076347867097419, 1.0] ,
(4, 16) : [-3.9314287799999996, 1.7519693918559192, 1.0] ,
(4, 17) : [-4.49116586, 1.8054031592694193, 1.0] ,
(4, 18) : [-0.011498579999999998, 4.301825308726518, 1.0] ,
(4, 19) : [-0.38783979, 3.3409701232276836, 1.0] ,
(4, 20) : [0.17695666000000002, 2.680262022172459, 1.0] ,
(5, 5) : [-3.6984473500000004, 1.6173197620755146, 1.0] ,
(5, 6) : [-5.36194871, 1.4939111420362323, 1.0] ,
(5, 7) : [-5.184545709999998, 1.2692468317864734, 1.0] ,
(5, 8) : [-9.519191779999998, 1.2152934491307028, 1.0] ,
(5, 9) : [-8.46137761, 1.2784613420827398, 1.0] ,
(5, 10) : [-0.06813073000000003, 3.092472793865776, 1.0] ,
(5, 11) : [-1.15930389, 2.482115409786579, 1.0] ,
(5, 12) : [-0.9447974399999999, 2.4162455175747355, 1.0] ,
(5, 13) : [-2.65547609, 2.040563697413046, 1.0] ,
(5, 14) : [-3.96431977, 1.9236156268859952, 1.0] ,
(5, 15) : [-4.0079606299999995, 1.7507569562906216, 1.0] ,
(5, 16) : [-6.62866308, 1.6215286455379074, 1.0] ,
(5, 17) : [-5.78044899, 1.727287667848063, 1.0] ,
(5, 18) : [-0.08027726000000004, 3.453986438421553, 1.0] ,
(5, 19) : [-1.21611198, 2.845291823117622, 1.0] ,
(5, 20) : [-1.57838947, 2.2922999617851065, 1.0] ,
(6, 6) : [-7.014413329999999, 1.3135526914440852, 1.0] ,
(6, 7) : [-8.66583655, 1.1761144598634947, 1.0] ,
(6, 8) : [-12.00687507, 1.143759750778108, 1.0] ,
(6, 9) : [-6.520291650000001, 1.2979988751921165, 1.0] ,
(6, 10) : [-0.12866449000000008, 2.8188260867779693, 1.0] ,
(6, 11) : [-2.28399021, 2.2554072795838893, 1.0] ,
(6, 12) : [-2.0375320500000003, 2.0920402474139927, 1.0] ,
(6, 13) : [-3.96831862, 1.9716973573041072, 1.0] ,
(6, 14) : [-4.88997186, 1.7278592446145609, 1.0] ,
(6, 15) : [-6.03596128, 1.5635395845005013, 1.0] ,
(6, 16) : [-8.09049026, 1.5430494234469614, 1.0] ,
(6, 17) : [-4.93281393, 1.651597047557303, 1.0] ,
(6, 18) : [-0.22569684999999992, 2.405108430882067, 1.0] ,
(6, 19) : [-2.4839157500000004, 2.5325873703191366, 1.0] ,
(6, 20) : [-2.8373892500000006, 2.2822021055769794, 1.0] ,
(7, 7) : [-10.379019409999998, 1.1176404245999694, 1.0] ,
(7, 8) : [-7.6079434899999985, 1.1705199357550473, 1.0] ,
(7, 9) : [-4.39031469, 1.3369180568381893, 1.0] ,
(7, 10) : [-0.0013385799999997058, 3.0286986831310907, 1.0] ,
(7, 11) : [-1.4614674399999996, 2.263374713298706, 1.0] ,
(7, 12) : [-1.1465495099999998, 2.082704493561196, 1.0] ,
(7, 13) : [-2.9895327099999998, 1.9509473886294322, 1.0] ,
(7, 14) : [-5.228592499999999, 1.5804444003823734, 1.0] ,
(7, 15) : [-6.777272179999999, 1.497825576861338, 1.0] ,
(7, 16) : [-5.7779805699999995, 1.5001811659596316, 1.0] ,
(7, 17) : [-3.7167633199999996, 1.609300366836471, 1.0] ,
(7, 18) : [2.45429307, 1.9349085981513443, 1.0] ,
(7, 19) : [-1.4746745600000002, 2.5636603618069222, 1.0] ,
(7, 20) : [-1.88054472, 2.234276259731549, 1.0] ,
(8, 8) : [-6.7836560299999995, 1.2355930845954102, 1.0] ,
(8, 9) : [-3.52346556, 1.366882535809131, 1.0] ,
(8, 10) : [1.4900372300000002, 2.8056798211485217, 1.0] ,
(8, 11) : [-3.1047057099999997, 2.076902123355841, 1.0] ,
(8, 12) : [-3.3138843000000002, 1.7511380074682865, 1.0] ,
(8, 13) : [-6.1643165, 1.6358180647003504, 1.0] ,
(8, 14) : [-8.8908866, 1.5306306191566925, 1.0] ,
(8, 15) : [-7.017157839999999, 1.4939631035604592, 1.0] ,
(8, 16) : [-6.6844801999999985, 1.495937641481088, 1.0] ,
(8, 17) : [-3.9325166499999997, 1.5701386980773386, 1.0] ,
(8, 18) : [0.8664759200000001, 1.7247415531609367, 1.0] ,
(8, 19) : [-3.29448433, 2.3267158113315, 1.0] ,
(8, 20) : [-4.94375649, 2.109118268376622, 1.0] ,
(9, 9) : [-2.70267164, 1.4231915075631951, 1.0] ,
(9, 10) : [-0.19061389000000006, 3.0966816773281685, 1.0] ,
(9, 11) : [-5.22795068, 1.9663972818329463, 1.0] ,
(9, 12) : [-5.03113631, 1.7755079623307806, 1.0] ,
(9, 13) : [-7.44152146, 1.6870694480963138, 1.0] ,
(9, 14) : [-6.57130862, 1.6365282055314536, 1.0] ,
(9, 15) : [-5.368563140000001, 1.61757956969665, 1.0] ,
(9, 16) : [-4.48477388, 1.6189305693265539, 1.0] ,
(9, 17) : [-3.47834617, 1.6476133306998944, 1.0] ,
(9, 18) : [-0.42079365999999996, 2.2701123909401493, 1.0] ,
(9, 19) : [-5.531137220000001, 2.18690467014454, 1.0] ,
(9, 20) : [-6.03986501, 2.1451968866982813, 1.0] ,
(10, 10) : [-0.0005707999999999998, 4.288523158524389, 1.0] ,
(10, 11) : [-0.0016775300000000104, 3.711888803668558, 1.0] ,
(10, 12) : [0.035269989999999994, 3.065418160251551, 1.0] ,
(10, 13) : [-0.04571984000000001, 3.277940794340252, 1.0] ,
(10, 14) : [-0.05972670000000002, 3.0685704927213258, 1.0] ,
(10, 15) : [-0.004846879999999887, 3.9298154362768747, 1.0] ,
(10, 16) : [0.82839733, 3.306086619963246, 1.0] ,
(10, 17) : [-0.1097552, 2.850453334524177, 1.0] ,
(10, 18) : [-0.008726260000000003, 3.5545839493251523, 1.0] ,
(10, 19) : [-0.005668240000000019, 4.090757597316173, 1.0] ,
(10, 20) : [-0.009714690000000005, 4.159502693868583, 1.0] ,
(11, 11) : [-0.02050432000000002, 4.346616142610249, 1.0] ,
(11, 12) : [-0.22623954, 3.4171457177445625, 1.0] ,
(11, 13) : [-0.7148464700000001, 2.966725905236276, 1.0] ,
(11, 14) : [-1.4993745900000004, 2.711213770103715, 1.0] ,
(11, 15) : [-1.46775193, 2.6644657188074308, 1.0] ,
(11, 16) : [-2.69820485, 2.5048745573980344, 1.0] ,
(11, 17) : [-4.16906119, 2.3750919903868986, 1.0] ,
(11, 18) : [0.0039716599999999914, 3.9192326058426286, 1.0] ,
(11, 19) : [-0.5719672900000001, 3.500007028378657, 1.0] ,
(11, 20) : [-0.34363384, 3.650591525588696, 1.0] ,
(12, 12) : [-0.13768366, 3.5538391674778977, 1.0] ,
(12, 13) : [-0.62470432, 2.865435574009648, 1.0] ,
(12, 14) : [-1.3156748, 2.53996590675938, 1.0] ,
(12, 15) : [-0.9101048999999999, 2.5443479953025294, 1.0] ,
(12, 16) : [-2.55694238, 2.1515015516378324, 1.0] ,
(12, 17) : [-3.45796, 2.2217362118847506, 1.0] ,
(12, 18) : [-0.006047110000000001, 4.106068926455083, 1.0] ,
(12, 19) : [-0.18932804, 3.8155347239934803, 1.0] ,
(12, 20) : [-0.19391302, 3.7679726088176384, 1.0] ,
(13, 13) : [-1.8599383499999997, 2.4900135614690937, 1.0] ,
(13, 14) : [-2.89504646, 2.430604218769481, 1.0] ,
(13, 15) : [-2.53466454, 2.2186704819553538, 1.0] ,
(13, 16) : [-4.69121422, 2.0464699906668558, 1.0] ,
(13, 17) : [-5.428191109999999, 2.1582219087711993, 1.0] ,
(13, 18) : [-0.05635430999999999, 3.7125469829754345, 1.0] ,
(13, 19) : [-0.7750131800000001, 3.3829723553112285, 1.0] ,
(13, 20) : [-0.8993533199999999, 3.0928538450434417, 1.0] ,
(14, 14) : [-3.6132131599999995, 2.282461913198115, 1.0] ,
(14, 15) : [-3.944880559999999, 2.0928196702773985, 1.0] ,
(14, 16) : [-6.67652414, 1.9472754409173856, 1.0] ,
(14, 17) : [-4.696822759999999, 2.0799158917610105, 1.0] ,
(14, 18) : [-0.10355210999999995, 3.0357481299178954, 1.0] ,
(14, 19) : [-1.69333926, 3.0462270373036873, 1.0] ,
(14, 20) : [-1.9085613400000003, 2.74365508172948, 1.0] ,
(15, 15) : [-5.27342455, 1.8986740952570034, 1.0] ,
(15, 16) : [-5.06603897, 1.9073516698029231, 1.0] ,
(15, 17) : [-3.8468736599999995, 2.0227062535870104, 1.0] ,
(15, 18) : [-0.004008839999999871, 3.9612694789423255, 1.0] ,
(15, 19) : [-1.4733755099999997, 3.0316778105201085, 1.0] ,
(15, 20) : [-1.72878825, 2.495157752367573, 1.0] ,
(16, 16) : [-5.375450689999999, 1.899522800152712, 1.0] ,
(16, 17) : [-3.59164903, 1.9762006894037862, 1.0] ,
(16, 18) : [0.93465713, 2.2390740404685148, 1.0] ,
(16, 19) : [-2.8429371100000003, 2.8243859698702654, 1.0] ,
(16, 20) : [-4.2476479199999995, 2.305515509446857, 1.0] ,
(17, 17) : [-3.05883707, 1.9932960308744911, 1.0] ,
(17, 18) : [-0.13947026999999998, 2.8504879755403287, 1.0] ,
(17, 19) : [-4.41424785, 2.682860098383812, 1.0] ,
(17, 20) : [-4.5597242399999995, 2.426499258355543, 1.0] ,
(18, 18) : [-0.0018171199999999985, 3.568024663591887, 1.0] ,
(18, 19) : [-0.003421970000000024, 7.54167830580833, 1.0] ,
(18, 20) : [-0.012286860000000004, 4.533642988811537, 1.0] ,
(19, 19) : [-0.5790672600000001, 3.935479242417624, 1.0] ,
(19, 20) : [-0.270541, 4.158896476085935, 1.0] ,
(20, 20) : [-0.26480296999999997, 4.137245840991324, 1.0] ,
}