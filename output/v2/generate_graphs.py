from bokeh.io import output_file, show
from bokeh.plotting import figure

for file_number in ["1","2"]:

  f=open("log."+file_number+".fired")
  lines=f.readlines()
  f.close()

  time_dict={}
  for l in lines:
    clock=l.split()[-1]
    neuron=l.split('[')[1].split(']')[0]
    if time_dict.has_key(clock):
      time_dict[clock].append(neuron)
    else:
      time_dict[clock]=[neuron]

  clocks=[]
  neurons=[]
  for t in time_dict.keys():
    for i in time_dict[t]:
      clocks.append(t)
      neurons.append(i)

  # output to static HTML file
  output_file("fired_neurons_"+file_number+".html", title="Neurons firing on each clock")

  # create a new plot with default tools, using figure
  p = figure(plot_width=600, plot_height=400)

  # add a circle renderer with a size, color, and alpha
  p.circle(clocks, neurons, size=10, line_color="navy", fill_color="orange", fill_alpha=0.5)

  show(p)
