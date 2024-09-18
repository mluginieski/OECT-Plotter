import os
import matplotlib.pyplot as plt
import Settings as S

class Plotter:
    
    def __init__(self, xlabel, ylabel, xunit='', yunit='', tiks_size=S.tiksSize, font_size=S.fontSize, fig_size=S.figSize):
       
        self.xlabel = xlabel + xunit
        self.ylabel = ylabel + yunit
        self.tiks_size = tiks_size
        self.font_size = font_size
        self.font_dict = {'fontsize': self.font_size}
        self.fig_size= fig_size
        self.folder = None
        self.plots = []  # List to hold plot data

    def set_folder(self, folder_name):
        self.folder = folder_name
        if not os.path.exists(self.folder):
            os.makedirs(self.folder) 

    def add_plot(self, x, y, label=None, plot_type='line', color=None, annotate_text=None, annotate_pos=(0,0), annotate_cords='data'):
        """Add data to be plotted along with optional annotation."""
        self.plots.append({
            'x': x,
            'y': y,
            'label': label,
            'plot_type': plot_type,  # 'line', 'scatter', or 'both'
            'color' : color,
            'annotate_text': annotate_text,
            'annotate_pos': annotate_pos,
            'annotate_cords': annotate_cords
        })

    def plot(self, xlog_scale=False, ylog_scale=False, legend_title=None, ylim=None, save_name=None, close=True):
        """Generate the plot with all added datasets."""
        
        plt.figure(figsize=self.fig_size)
        has_labels = False  # Flag to track if any labels are provided

        for plot_data in self.plots:
            x              = plot_data['x']
            y              = plot_data['y']
            label          = plot_data['label']
            color          = plot_data['color']
            plot_type      = plot_data['plot_type']
            annotate_text  = plot_data['annotate_text']
            annotate_pos   = plot_data['annotate_pos']
            annotate_cords = plot_data['annotate_cords']

            # Plot based on the type specified
            if plot_type == 'scatter':
                plt.scatter(x, y, marker='^', color=color, s=50, label=label)
            elif plot_type == 'line':
                plt.plot(x, y, label=label, color=color)
            elif plot_type == 'line_scatter':
                plt.plot(x, y, '-o', color=color, label=label)
            elif plot_type == 'both':
                plt.scatter(x, y, label=label)
                plt.plot(x, y, label=None, color=color)  # Avoid duplicating the label in the legend
            
            if annotate_text:  # Annotate if text is provided for this plot
                props = dict(boxstyle='round', facecolor='white', alpha=0.5)
                plt.annotate(annotate_text, annotate_pos, xycoords=annotate_cords, fontsize=self.font_size, verticalalignment='top', bbox=props)
            
            if label:  # Check if the plot has a label
                has_labels = True

        if xlog_scale:
            plt.xscale('log')
        
        if ylog_scale:
            plt.yscale('log')
        
        if ylim:
            plt.ylim(ylim)
        
        plt.xlabel(self.xlabel, fontdict=self.font_dict)
        plt.ylabel(self.ylabel, fontdict=self.font_dict)
        plt.xticks(fontsize=self.tiks_size)
        plt.yticks(fontsize=self.tiks_size)
        plt.grid(visible=True, which='both', color='gray', alpha=0.1)
        plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        
        # Only add a legend if there are any labels
        if has_labels:
            plt.legend(title=legend_title, title_fontsize=self.tiks_size, fontsize=self.tiks_size, loc='best')
        
        if save_name and self.folder:
            plt.tight_layout()
            plt.savefig(f'{self.folder}/{save_name}.png')
        
        if close:
            plt.close('all')
        
        # Clear the plots after drawing to avoid overlapping in future plots
        self.plots.clear()