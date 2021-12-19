import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from .helper_functions import *
sns.set_theme()
sns.set_style("whitegrid")

# Read config file
config = read_config()

def get_default_values():
    """Get default parameters from config file"""

    figsize = (config["plotting"]["figsize_x"], config["plotting"]["figsize_y"])
    title_fontsize = config["plotting"]["title_fontsize"]
    label_fontsize = config["plotting"]["label_fontsize"]
    tick_fontsize = config["plotting"]["tick_fontsize"]
    cmap = config["plotting"]["tick_fontsize"]
    save_fig = config["plotting"]["save_fig"]

    return figsize, title_fontsize, label_fontsize, tick_fontsize, cmap, save_fig

def single_plot_decorator(func):
    """
    Decorator function which wraps around plotting functions. It handles figure size and font sizes by reading a configuration file.
    
    Args:
        func: Function to wrap around.

    Returns:
        wrap (func): Wrapped function.
    """
    
    def wrap(*args, **kwargs):
        figsize, title_fontsize, label_fontsize, tick_fontsize, cmap, save_fig = get_default_values()

        fig, ax = plt.subplots(figsize=figsize)

        kwargs["fig"]=fig
        kwargs["ax"]=ax
        kwargs["cmap"]=cmap
        func(*args, **kwargs)

        ax.xaxis.label.set_size(label_fontsize)
        ax.yaxis.label.set_size(label_fontsize)
        ax.title.set_size(title_fontsize)

        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        plt.show()
        return 

    return wrap

@single_plot_decorator
def plot_distribution(df, target_col, x_label, y_label, title, cmap=None, fig=None, ax=None):
    """
    Function to plot distribution of values in a histogram.
    
    Args:
        df (pd DataFrame): Pandas dataframe containing data to plot
        target_col (str): Column name of target variable
        x_label (str): Label text for x axis
        y_label (str): Label text for y axis
        title (str): Text for figure title
        cmap (str): colormap, supplied by decorator function
        fig: matplotlib figure object, supplied by decorator function
        ax: matplotlib ax object, supplied by decorator function
    """

    sns.histplot(df[target_col], ax=ax, kde=True, stat="count", log_scale=(False, True))

    if title is not None:
        ax.set_title(title)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    return

@single_plot_decorator
def barplot_classes(x, y, x_label, y_label, xlim=None, ylim=(0, 1), legend_title=None, title=None, s=10, cmap=None, fig=None, ax=None):
    sns.barplot(x=x, y=y, log=False)

    if title is not None:
        ax.set_title(title)

    if y_label is not None:
        ax.set_ylabel(y_label)

    ax.set_xticklabels([camel_case_split(str(s)) for s in x])

    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=config["plotting"]["legend_fontsize"], title=legend_title, title_fontsize=config["plotting"]["legend_fontsize"])
    plt.xticks(rotation=90)

@single_plot_decorator
def barplot_best_models(dataframes, x_col, y_col, hue_col, x_label, y_label, xlim=None, ylim=(0, 1), legend_title=None, title=None, s=10, cmap=None, fig=None, ax=None):
    """
    Function to create scatterplot of expected value and real value of target, where the color of each marker is defined by the value_col.
    
    Args:
        dataframes (list of pd DataFrame): Pandas dataframes containing data to plot
        x_col (str): Column name of x-axis variable
        y_col (str): Column name of y-axis variable
        value_col (str): Column name of variable defining hue
        x_label (str): Label text for x axis
        y_label (str): Label text for y axis
        s (int): markersize
        title (str): Text for figure title
        cmap (str): colormap, supplied by decorator function
        fig: matplotlib figure object, supplied by decorator function
        ax: matplotlib ax object, supplied by decorator function
    """

    plot_data = []
    for df in dataframes:
        plot_data.append(df.iloc[np.argmax(df["test_{}".format(y_col)])])

    plot_data = pd.DataFrame(plot_data)

    x = np.arange(len(plot_data)) 
    width = 0.35
    rects1 = ax.bar(x - width/2, plot_data["train_{}".format(y_col)], width, label='Train')
    rects2 = ax.bar(x + width/2, plot_data["test_{}".format(y_col)], width, label='Test')

    ax.set_xticks(x)
    ax.set_xticklabels(plot_data["model_name"])

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if title is not None:
        ax.set_title(title)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=config["plotting"]["legend_fontsize"], title=legend_title, title_fontsize=config["plotting"]["legend_fontsize"])
    # plt.xticks(rotation=90)
    # sns.histplot(
    #     df, x=x_col, y=y_col, hue=value_col, legend=False, cmap=cmap
    # )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.tight_layout()

    return



@single_plot_decorator
def plot_scatter(df, x_col, y_col, hue_col, x_label, y_label, xlim=None, ylim=(0, 1), legend_title=None, title=None, s=10, cmap=None, fig=None, ax=None):
    """
    Function to create scatterplot of expected value and real value of target, where the color of each marker is defined by the value_col.
    
    Args:
        df (pd DataFrame): Pandas dataframe containing data to plot
        x_col (str): Column name of x-axis variable
        y_col (str): Column name of y-axis variable
        value_col (str): Column name of variable defining hue
        x_label (str): Label text for x axis
        y_label (str): Label text for y axis
        s (int): markersize
        title (str): Text for figure title
        cmap (str): colormap, supplied by decorator function
        fig: matplotlib figure object, supplied by decorator function
        ax: matplotlib ax object, supplied by decorator function
    """
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, s=s, ax=ax, palette="colorblind")

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if title is not None:
        ax.set_title(title)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=config["plotting"]["legend_fontsize"], title=legend_title, title_fontsize=config["plotting"]["legend_fontsize"])
    # sns.histplot(
    #     df, x=x_col, y=y_col, hue=value_col, legend=False, cmap=cmap
    # )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    return

@single_plot_decorator
def plot_error_scatter_grouped(df, x_col, y_col, value_col, x_label, y_label, s=10, cmap=None, fig=None, ax=None):
    """
    Function to create scatterplot of expected value and real value of target, where the color of each marker is defined by the value_col.
    
    Args:
        df (pd DataFrame): Pandas dataframe containing data to plot
        x_col (str): Column name of x-axis variable
        y_col (str): Column name of y-axis variable
        value_col (str): Column name of variable defining hue
        x_label (str): Label text for x axis
        y_label (str): Label text for y axis
        s (int): markersize
        title (str): Text for figure title
        cmap (str): colormap, supplied by decorator function
        fig: matplotlib figure object, supplied by decorator function
        ax: matplotlib ax object, supplied by decorator function
    """
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=value_col, s=s, ax=ax)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Draw a line of x=y 
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    ax.plot(lims, lims, 'r', linestyle="--")

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # sns.histplot(
    #     df, x=x_col, y=y_col, hue=value_col, legend=False, cmap=cmap
    # )


    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    return

@single_plot_decorator
def plot_error_scatter_individual(df, x_col, y_col, value_col, x_label, y_label, title=None, s=10, cmap=None, fig=None, ax=None):
    """
    Function to create scatterplot of expected value and real value of target, for a subset of data.
    
    Args:
        df (pd DataFrame): Pandas dataframe containing data to plot
        x_col (str): Column name of x-axis variable
        y_col (str): Column name of y-axis variable
        value_col (str): Column name of variable defining hue
        x_label (str): Label text for x axis
        y_label (str): Label text for y axis
        s (int): markersize
        title (str): Text for figure title
        cmap (str): colormap, supplied by decorator function
        fig: matplotlib figure object, supplied by decorator function
        ax: matplotlib ax object, supplied by decorator function
    """
    sns.scatterplot(data=df, x=x_col, y=y_col, s=s, ax=ax)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Draw a line of x=y 
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    ax.plot(lims, lims, 'r', linestyle="--")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_title(title)

    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    return

def plot_error_scatter(df, x_col, y_col, value_col, x_label, y_label, s=10):
    """
    Function to create multiple scatterplots of expected value and real value of target.
    
    Args:
        df (pd DataFrame): Pandas dataframe containing data to plot
        x_col (str): Column name of x-axis variable
        y_col (str): Column name of y-axis variable
        value_col (str): Column name of variable defining hue
        x_label (str): Label text for x axis
        y_label (str): Label text for y axis
        s (int): markersize
    """
    plot_error_scatter_grouped(df, x_col, y_col, value_col, x_label, y_label, s=s, cmap=None)

    for group in df[value_col].unique():
        plot_error_scatter_individual(df[df[value_col]==group],  x_col, y_col, value_col, x_label, y_label, s=s, title=group)

    return


@single_plot_decorator
def plot_simulation_in_situ_correlation_individual(in_situ_df, simulation_df, feature_cols, target_cols, x_label, y_label, title=None, s=10, cmap=None, fig=None, ax=None):
    """
    Function to visualize correlation between features and target variable for both the in situ dataset and the simulation dataset.
    
    Args:
        in_situ_df (pd DataFrame): Pandas dataframe containing in situ data
        simulation_df (pd DataFrame): Pandas dataframe containing simulation data
        x_col (str): Column name of x-axis variable
        y_col (str): Column name of y-axis variable
        feature_cols (list of str): Column names of features
        target_cols (tuple of str): Column names of target variables
        x_label (str): Label text for x axis
        y_label (str): Label text for y axis
        s (int): markersize
        title (str): Text for figure title
        cmap (str): colormap, supplied by decorator function
        fig: matplotlib figure object, supplied by decorator function
        ax: matplotlib ax object, supplied by decorator function
    """

    x1 = in_situ_df[feature_cols].corrwith(in_situ_df[target_cols[0]])
    x2 = simulation_df[feature_cols].corrwith(simulation_df[target_cols[1]])

    ax.plot(feature_cols, x1, label="in situ")
    ax.plot(feature_cols, x2, label="simulation", color="red")
    ax.axhline(0,0, color="black", linestyle="--")

    ax.set_ylim(-1, 1)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    ax.legend()

    return


def plot_simulation_in_situ_correlation(in_situ_df, simulation_df, feature_cols, target_cols, group_col, x_label, y_label, title=None, s=10, cmap=None, fig=None, ax=None):
    """
    Function to create multiple plots of correlation between features and target variable for both the in situ dataset and the simulation dataset.
    
    Args:
        in_situ_df (pd DataFrame): Pandas dataframe containing in situ data
        simulation_df (pd DataFrame): Pandas dataframe containing simulation data
        x_col (str): Column name of x-axis variable
        y_col (str): Column name of y-axis variable
        feature_cols (list of str): Column names of features
        target_cols (tuple of str): Column names of target variables
        x_label (str): Label text for x axis
        y_label (str): Label text for y axis
        s (int): markersize
        title (str): Text for figure title
        cmap (str): colormap, supplied by decorator function
        fig: matplotlib figure object, supplied by decorator function
        ax: matplotlib ax object, supplied by decorator function
    """
    plot_simulation_in_situ_correlation_individual(in_situ_df, simulation_df, feature_cols, target_cols, x_label, y_label, title)

    for group in in_situ_df[group_col].unique():
        plot_simulation_in_situ_correlation_individual(in_situ_df[in_situ_df[group_col]==group], simulation_df, feature_cols, target_cols, x_label, y_label, title=group)

    return