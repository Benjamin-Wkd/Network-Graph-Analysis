from hello import *  


if __name__ == "__main__":
  nwin, N, theta, so, spec, Sharing, RippleDensity_by_win, flat_adj = load_from_matlab('File4', k=2, weight=True)
  starts, ends = delimitation_SO(spec,so)
  print(starts,ends)
  Adjs = dict_adj(N, flat_adj)
  COS = cosine_similarity(Adjs)
  # idx where substates == 4
  idx = find(Sharing,lambda e:e==4)
  # plots
  fig = plot_liquidity(spec, so, COS, Sharing, subs =[1,2,3,4])
  plt.show()