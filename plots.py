
def plot_liquidity(spec, so, COS, Sharing, neurons ='all', subs =[1,2,3,4]):
    """
    - neuron : 'all' : display the average liquidity over all neurons
                [x,y,z] (list of int) : display the average liquidity over selected neurons
    """
    starts, ends = delimitation_SO(spec, so)
    nb_periods = len(starts)
    fig, ax = plt.subplots(figsize=(15,6))
    for i in range(nb_periods):
        if i==0:
            ax.axvspan(xmin=starts[i], xmax = ends[i] , facecolor='yellow', alpha=0.3)  # legend appears only once
        else : 
            ax.axvspan(xmin=starts[i], xmax = ends[i] , facecolor='yellow', alpha=0.3, label='_nolegend_')
    legend = ['SWS']
    for i in range(len(subs)):
        idx_sub = find(Sharing, lambda e:e==subs[i])
        idx_sub.pop()
        cos_sub = COS[idx_sub,:]
        if neurons=='all':
            cos_sub = np.nanmean(cos_sub,axis=1)  #average of liquidity over all neurons
            title = 'Average liquidity of all neurons'
        else :
            cos_sub = np.nanmean(cos_sub[:,neurons],axis=1)
            title = 'Average liquidity of neurons n° {}'.format(neurons)
        ax[0].scatter(idx_sub, cos_sub,s=5)
        legend.append('sub'+str(i))
    ax[0].set_ylim([.6,1])
    ax[0].legend(legend)
    ax[0].set_title(title)
    ## 
    # Plot smooth black curves for each neuron
    if neurons=='all': 
        neurons = range(N)   
    nwin = len(Sharing)
    smooth_step = 100 # 1 point is the average coreness of 100 time-windows
    bins = range(0,nwin,smooth_step)  #smooth curve
    for n in neurons:
        smooth_liq = []
        for bin in bins:
            if bin+smooth_step>nwin:
                smooth_liq.append(np.nanmean(COS[bin:,n],axis=0))
            else:
                smooth_liq.append(np.nanmean(COS[bin:bin+smooth_step,n],axis=0))
        if n in range(17,18):  # n=17 # n=8  #n=29
            color = 'fuchsia'
        else: 
            color='k'
        ax[1].plot(range(0,nwin,smooth_step), smooth_liq, color=color, label='_nolegend_' )
    ax[1].set_ylim([.6,1])
    ax[1].set_title('liquidity for each neuron')
    ax[1].set_ylabel('Liquidity')
    
def plot_coreness(file, spec, so, CORE, Sharing, neurons = 'all', subs =[1,2,3,4]):
    """
    - CORE : array (nwin,N) coreness values for each neuron
    - neuron : 'all' : display the average coreness over all neurons
                [x,y,z] (list of int) : display the average coreness over selected neurons
    """
    _,N = np.shape(CORE)
    starts, ends = delimitation_SO(spec, so)
    nb_periods = len(starts)
    fig, ax = plt.subplots(figsize=(15,6))
    for i in range(nb_periods):
        if i==0:
            ax.axvspan(xmin=starts[i], xmax = ends[i] , facecolor='yellow', alpha=0.3)  # legend appears only once
        else : 
            ax.axvspan(xmin=starts[i], xmax = ends[i] , facecolor='yellow', alpha=0.3, label='_nolegend_')
    legend = ['SWS']
    # for i in range(len(subs)):
    #     idx_sub = find(Sharing, lambda e:e==subs[i]) # index when sharing = subs i
    #     core_sub = CORE[idx_sub,:]  
    #     if neurons=='all':
    #         core_sub = np.nanmean(core_sub,axis=1)  #average of liquidity over all neurons
    #         title = 'Average coreness of all neurons'
    #     else :
    #         print(np.shape(core_sub[:,neurons]))
    #         core_sub = np.nanmean(core_sub[:,neurons],axis=1)
    #         title = 'Average coreness of neurons n° {}'.format(neurons)
    #     ax[0].scatter(idx_sub, core_sub,s=5)
    #     legend.append('sub'+str(i))
    # ax[0].legend(legend)
    # ax[0].set_title(title)
    ## 
    # Plot smooth black curves for each neuron$
    hubs_rip, hubs_SO, both, not_hubs = Get_Hubs( "HPC", file=file, F=4, St=3, Sh=4, selected_subsate=4) # Get Hubs
    if neurons=='all': 
        neurons = range(N)   
    nwin = len(Sharing)
    smooth_step = 100 # 1 point is the average coreness of 100 time-windows
    bins = range(0,nwin,smooth_step)  #smooth curve
    for type, hub_list in enumerate([hubs_rip, hubs_SO, both, not_hubs]):
        smooth_core = []
        avg_hubs_core = np.nanmean(CORE[:,hub_list],axis=1)
        for bin in bins:
            if bin+smooth_step>nwin:
                smooth_core.append(np.nanmean(avg_hubs_core[bin:],axis=0))
            else:
                smooth_core.append(np.nanmean(avg_hubs_core[bin:bin+smooth_step],axis=0))
        if type == 0 :  # n=17 # n=8  #n=29
            color = 'red'
            print("Hubs only in ripples: ", hub_list)
        elif type == 1: 
            color = 'green'
            print("Hubs only in SO: ", hub_list)
        elif type == 2 : 
            color = 'orange'
            print("Hubs in both: ", hub_list)
        else :
            color = 'b'
            print("Not Hubs: ", hub_list)
        ax.plot(range(0,nwin,smooth_step), smooth_core, color=color )
    ax.set_title('Coreness for types of hubs')
    ax.set_ylabel('Coreness')
    ax.legend(['hubs in ripples only', 'hubs in SO only', 'hubs in both', 'not hubs'], loc='upper right')
    
def plot_centralisation(spec, so, C, Sharing, subs =[1,2,3,4]):
    """
    - C : (list) nwin - value of cp-centralisation between 0 and 1 for each time-window
    """
    nwin = len(C)
    C = np.array(C)
    starts, ends = delimitation_SO(spec, so)
    nb_periods = len(starts)
    fig, ax = plt.subplots(figsize=(15,6))
    for i in range(nb_periods):
        if i==0:
            ax.axvspan(xmin=starts[i], xmax = ends[i] , facecolor='yellow', alpha=0.3)  # legend appears only once
        else : 
            ax.axvspan(xmin=starts[i], xmax = ends[i] , facecolor='yellow', alpha=0.3, label='_nolegend_')
    legend = ['cp-centralisation']
    legend.append('SWS')
    for i in range(len(subs)):
        idx_sub = find(Sharing, lambda e:e==subs[i]) # index when sharing = subs i
        C_sub = C[idx_sub]  
        ax.scatter(idx_sub, C_sub, s=5)
        legend.append('sub'+str(i+1))
    ## 
    # Plot smooth black curve
    nwin = len(Sharing)
    smooth_step = 100 # 1 point is the average coreness of 100 time-windows
    bins = range(0,nwin,smooth_step)  #smooth curve
    smooth_C = []
    for bin in bins:
        if bin+smooth_step>nwin:
            smooth_C.append(np.mean(C[bin:]))
        else:
            smooth_C.append(np.mean(C[bin:bin+smooth_step]))
    ax.plot(range(0,nwin,smooth_step), smooth_C, 'k')
    ax.legend(legend)
    ax.set_title('cp-centralisation')