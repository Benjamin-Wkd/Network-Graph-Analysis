from os.path import dirname, join as pjoin
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import networkx as nx   # https://github.com/networkx/networkx
import cpnet            # https://github.com/skojaku/core-periphery-detection
from tqdm import tqdm   # to show progressbar
from scipy.integrate import quad
from scipy import signal
from numpy.polynomial.polynomial import Polynomial as poly

def load_from_matlab(file, k, weight=True):
    """
    Input : 
    - file : (int) number of the file 
    - k = nb of clusters+2  (int between 1 and 8)
    - weighted graph (default) or not
    Output : 
    - nwin : number of time windows (int)
    - N : nb of neuron (int) 
    - theta, so values (int 1 & 2)
    - spec : Spectogram of the file (nwin,) list
    - Sharing : (nwin,) array with n° substate
    - RippleDensity_by_win : (nwin,) list
    - flat_adj : contains for each row the flattened adjency matrix for one period of time :  (nwin, N*(N-1)) array
    """
    File = 'File'+str(file)
    data_dir = '/mnt/c/Users/waked/OneDrive - CentraleSupelec/Documents/Stages INS/Script/Data/'
    mat_clusters = pjoin(data_dir, 'Pascale data', 'ClusterData.C.Feat.HPC.REDO')
    ClusterData = sio.loadmat(mat_clusters)
    ClusterData = ClusterData['ClusterData']    
    ClusterData_file = ClusterData[File][0,0]
    spec = ClusterData_file['Spectrogram'][0,0]
    theta = ClusterData_file['IsTheta'][0,0] 
    theta = theta[0,0]
    so = 3-theta  # 2 if theta=1 and 1 if theta=2
    spec = spec['clust_feat'][0,0]
    spec = spec[:,0] 
    
    Sharing = ClusterData_file['Sharing'][0,0]
    Sharing = Sharing['clust'][0,0]
    if file==2 or file==3:
        Sharing = Sharing[k-1,:]  # Clusterization done only one time and not 3-times
    else:
        Sharing = Sharing[k-1,:,1]
    
    mat_RippleDensity =  pjoin(data_dir, 'useful outputs', 'AllRippleDensity_by_win')
    Ripples_win = sio.loadmat(mat_RippleDensity)
    AllRippleDensity_by_win = Ripples_win["AllRippleDensity_by_win"]
    Ripples_Pascale = AllRippleDensity_by_win['Pascale_data'][0,0]
    RippleDensity_by_win = Ripples_Pascale[File][0,0]
    RippleDensity_by_win = RippleDensity_by_win[0]
    
    mat_Features =  pjoin(data_dir, 'Pascale data', File, File+'_Features_HPC')
    Features = sio.loadmat(mat_Features)
    Features_Spikenet = Features['Features_Spikenet']
    nwin,N = np.shape(Features_Spikenet['SpikeDensity'][0,0])
    if weight: info = 'Graph_w'
    else: info = 'Graph'
    flat_adj = Features_Spikenet[info][0,0] #flattened adjency matrices 
    
    print("{} time-windows, {} neurons, so : {}, spec : {}, Sharing with {} clusters : {},  ".format(nwin, N, so,spec,k+2,Sharing))
    return nwin, N, theta, so, spec, Sharing, RippleDensity_by_win, flat_adj

def find(array, condition):
    """
    e>2 condition should for e.g be written:  lambda e: e>2 
    """
    return [i for i, elem in enumerate(array) if condition(elem)]

def delimitation_SO(spec,so):
    """
    
    """
    num_index = find(spec, lambda e: e==so)
    distance = np.diff(num_index) #distance between 2 SO index
    gaps = find(distance, lambda e: e>1) #find the gaps = zone of discontinuity (start of a new SO)
    starts = [] #assuming that max 6 blocks of SO
    ends = []
    starts.append(num_index[0])
    for i in range(len(gaps)): 
        starts.append(num_index[gaps[i]+1])
        ends.append(num_index[gaps[i]])
    ends.append(num_index[-1])
    return starts, ends

def dict_adj(N, flat_adj):
    """
    return a dictionnary of all Adj matrices for each time-window. 
    Keys are time-windows (start=0)
    Values are adjency matrices
    """
    def check_symmetric(a, rtol=1e-05, atol=1e-08):
        return np.allclose(a, a.T, rtol=rtol, atol=atol)
    
    nwin,_ = np.shape(flat_adj)
    Adjs = {} #dictionnary that stores adjency matrices for each window
    for t in tqdm(range(nwin)):  # Create a graph for each window of time
        flat_graph_t = flat_adj[t,:]
        flat_graph_t = np.insert(flat_graph_t,[N*i for i in range(N)],0)  #insert the value A(i,i) for each neuron : flattened adj matrix
        Adj_0 = np.reshape(flat_graph_t,(N,N)) # Adjency matrix
        # Checking the transformation
        c=0
        for i in range(N):
            if Adj_0[i,i] != 0:
                print("error for i=%d", i)
                c+=1
        if not(c==0 and check_symmetric(Adj_0)):
            print("error for win=%d", {t})
        Adjs[t] = Adj_0
    return Adjs

def Get_Hubs(region, file, F, St, Sh, selected_subsate):
  """
  - region : "HPC"
             "EC"
  - file : int 1 to 5
  - F/St/Sh : Number of substates for each feature
  - selected_subsate : (int) Chosen Sharing substate
  """  
  Hubs = sio.loadmat(r'/mnt/c/Users/waked/OneDrive - CentraleSupelec/Documents/Stages INS/Script/Data/useful outputs/Hubs_P')
  Index = sio.loadmat(r'/mnt/c/Users/waked/OneDrive - CentraleSupelec/Documents/Stages INS/Script/Data/useful outputs/Index_P')
  
  def clust2index(F,St,Sh):
    index = (F-3)*64+(St-3)*8+(Sh-3)
    return index
  index = clust2index(F,St,Sh)
  
  Hubs = Hubs['Hubs']  
  Hubs_R = Hubs['R'][0,0]
  Hubs_R = Hubs_R['HPC'][0,0]
  Hubs_R = Hubs_R['c'][0,0]
  Hubs_R = Hubs_R['neuron'][0,0]
  HR = Hubs_R[(file-1)*512:file*512]
  
  Hubs_SO = Hubs['SO'][0,0]
  Hubs_SO = Hubs_SO['HPC'][0,0]
  Hubs_SO = Hubs_SO['c'][0,0]
  Hubs_SO = Hubs_SO['neuron'][0,0]
  HSO = Hubs_SO[(file-1)*512:file*512]
  
  rip_n = HR[index,0]
  SO_n = HSO[index,0]
  hubs_rip = find(rip_n[:,F+St+selected_subsate-1], lambda e: e==1)
  hubs_SO = find(SO_n[:,F+St+selected_subsate-1], lambda e: e==1)
  both = []
  only_hubsRip = []
  only_hubsSO = []
  N,_ = np.shape(rip_n)
  not_hubs = list(range(N))
  for h in hubs_rip:
    not_hubs = np.delete(not_hubs, find(not_hubs, lambda e: e==h))
    if h in hubs_SO:
      both.append(h)
    else : 
      only_hubsRip.append(h)
  for h in hubs_SO:
    if not(h in hubs_rip):
      only_hubsSO.append(h)
      not_hubs = np.delete(not_hubs, find(not_hubs, lambda e: e==h))
      
  return only_hubsRip, only_hubsSO, both, not_hubs

def coreness(Adjs):
    """
     Parameters ---------
    - nwin; number of time-windows
    - N: number of neurons
    - Adjs: dic
            keys are the time of each windows and values are adjency matrix for that time.
     Returns ---------
    - CORE: array (time windows, nb of neuron)
          rows show the coreness for time window t for each neuron (column) 
    """
    nwin = len(Adjs) #nb of time windows
    N,_ = np.shape(Adjs[0]) #nb of neuron
    CORE = np.zeros((nwin,N))
    print("Computing coreness . . .")
    for time in tqdm(range(nwin)):
        G = nx.Graph(Adjs[time]) # transform adj matrix into nx graph object
        algo = cpnet.Rossa() #Load the algorithm
        algo.detect(G) # Give the network as an input
        coreness = algo.get_coreness()  # Get the coreness of nodes
        # pair = algo.get_pair_id()  # Get the group membership of nodes
        CORE[time,:] = list(coreness.values())
    return CORE    

def complete(n):  #all to all network
  x = range(1,n+1)
  y = n*[0]
  for i in range(n):
    y[i] = i/(n-1)
  new_series = poly.fit(x,y,deg=20)
  coef = new_series.convert().coef
  complete = poly(coef)
  return complete

def star(n):  #only one node connected to all others
  x = range(1,n+1)
  y = (n-1)*[0]
  y.append(1)
  new_series = poly.fit(x,y,deg=20)
  coef = new_series.convert().coef
  star = poly(coef)
  return star

def cp_profile(n, coreness):
  x = range(1,n+1)
  y = np.sort(coreness)
  new_series = poly.fit(x,y,deg=20)
  coef = new_series.convert().coef
  f = poly(coef)
  return f

def cp_centralisation(CORE,time2plot):
  """
  Parameters ---------
    - CORE: array (nb of time-windows, nb of neurons)
            for each time window, returns the coreness of each neuron (meaningful only if high cp-centralisation)
    - time2plot: (int) value between 0 and nwin to display cp-centralisation
  Returns ---------
    - C : list (nb time windows)
          cp-centralisation 
  """
  nwin,N = np.shape(CORE)
  C = nwin*[0]
  print("Computing cp-centralisation . . .")
  for time in tqdm(range(nwin)): 
    f_complete = complete(N)
    f_star = star(N)
    f = cp_profile(N,CORE[time,:])
    res, _= quad(f, 1, N)
    resC, _ = quad(f_complete, 1, N)
    resS,_ = quad(f_star, 1, N)
    C[time] = (resC-res)/(resC-resS)  #cp-centralisation between 0 (like complete) and 1 (like star)
  if time == time2plot:
    X = np.linspace(1,N,100)
    fig = plt.figure()
    plt.plot(f_complete(X))
    plt.plot(f(X),'green')
    plt.plot(f_star(X),'r')
    plt.xticks(range(1,N+1))
    plt.legend(['Complete : ' + str(round(resC,3)), 'Network : ' + str(round(res,3)), 'Star : ' + str(round(resS,3))])
  return C

def Jacccard_similarity(Adjs):
    """
    Parameters ---------
    - Adjs: dic
            keys are the time of each windows and values are adjency matrix for that time.
    Returns ---------
    - J : array (time windows-1, nb of neuron)
          rows show the Jaccard similarity for time window t and t+1 for each neuron 
          jaccard distance = 1-J
    """
    nwin = len(Adjs) #nb of time windows
    N,_ = np.shape(Adjs[0]) #nb of neuron
    J = np.zeros((nwin-1,N)) 
    for time in range(nwin-1):
        adj_t0 = Adjs[time]
        adj_t1 = Adjs[time+1]
        for i in range(N): # for each neuron
            ni_t0 = adj_t0[i,:]
            ni_t1 = adj_t1[i,:]
            num = (ni_t0*ni_t1).sum()
            den =  (ni_t0+ni_t1)
            den[den==2]=1
            den = den.sum()
            J[time,i] = num/den
    return J

def cosine_similarity(Adjs):
    """
    Parameters
    ---------
    - Adjs: dic
            keys are the time of each windows and values are adjency matrix for that time.
    Returns
    ---------
    - COS : array (time windows-1, nb of neuron)
          rows show the Jaccard similarity for time window t and t+1 for each neuron 
          jaccard distance = 1-J
    """
    nwin = len(Adjs) #nb of time windows
    N,_ = np.shape(Adjs[0]) #nb of neuron
    COS = np.zeros((nwin-1,N)) 
    print("Computing cosine similarity . . .")
    for time in tqdm(range(nwin-1)):
        adj_t0 = Adjs[time]
        adj_t1 = Adjs[time+1]
        for i in range(N): # for each neuron
            ni_t0 = adj_t0[i,:]
            ni_t1 = adj_t1[i,:]
            num = (ni_t0*ni_t1).sum()
            den =  np.sqrt(np.sum(ni_t0**2)*np.sum(ni_t1**2))
            COS[time,i] = num/den
    return COS

    
def correlation_ripples(spec, so, RippleDensity_by_win, Sharing, CORE, C, selected_neurons):
    subs = np.unique(Sharing)
    # for s in subs:
    C = np.array(C)
    x = RippleDensity_by_win[spec==so]
    yC= C[spec==so]
    CORE_so = CORE[spec==so, :]
    ycore = np.nanmean(CORE_so[:,selected_neurons],axis=1)
    corr_C0 = np.corrcoef(x, yC)
    corr_core0 = np.corrcoef(x,ycore )
    corr_C = signal.correlate(x, yC)
    corr_core = signal.correlate(x, ycore)
    return corr_C0[0,1], corr_core0[0,1], corr_C, corr_core
    