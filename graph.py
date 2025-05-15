import matplotlib.pyplot as plt
import numpy as np
from lib import *
from matplotlib.backends.backend_pdf import PdfPages

TITLE = {"obj": "Eye tracking", "pos": "Camera Position", "rot": "Camera Rotation", "bpm": "Heart rate", "trigger": "Button", "state": "State", "subTask": "Subtask", "warning": "Warning", "collision": "Collision"}
UI = ["VA", "VO", "AO", "NO"]
OBJ_LABEL = ["Robot", "Arrow", "TaskPanel", "Shelf", "Building", "None"]
OBJ_COLOR = {"Robot": "red", "Arrow": "blue", "TaskPanel": "yellow", "Shelf": "green", "Building": "grey", "None": "grey"}
PATH = "pic/{0}_{1:02d}_{2:01d}.{3}"
TYPE = "png"
DPI = 300
SAVE = False
FIGSIZE = (11.69, 8.27)
PPATH = "pdf/{0}.pdf"

def pie_plot(id, uiId, attr, target = None):
    data = load_subtask(id, uiId)
    for i, sub in enumerate(data):
        labels, counts = np.unique(sub[attr], return_counts=True)
        dic = dict(zip(labels, counts))
        print(dic)
        if target:
            target.reverse()
            value = [dic.get(label, 0) for label in target]
        else:
            value = dic.values()
            target = dic.keys()
        
        plt.figure(figsize=(7, 6))
        plt.pie(value, labels=target, autopct='%1.1f%%', startangle=90)
        plt.title('{0}: ({1:02d}, {2:01d}, {3})'.format(attr, id, uiId, i))
        # plt.savefig(PIC.format("pie_subtask", id, uiId))
        print(dic)
        plt.show()
        plt.close()

def barh_plot(userId, uiId, attr, target):
    data = load_subtask(userId, uiId)
    tasks = [ "{0}-{1}*".format(i, ROBOT_NUM[d["state"]])  if d["collision_flag"] else "{0}-{1} ".format(i, ROBOT_NUM[d["state"]]) for i, d in enumerate(data)]
    data = [ dict( [(label, count) for label, count in zip(*np.unique(sub[attr], return_counts=True))]) for sub in data ]
    data = [ [dic.get(label, 0) for label in target] for dic in data]
    data = np.array([ l/sum(l)*100 for l in data])
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.xaxis.set_visible(False)  # hide the x-axis
    ax.set_xlim(0, data.sum(axis=1).max())  # scale x-axis
    ax.set_ylim(-1, len(tasks) -0.5)

    for i, (colname, color) in enumerate(zip(target, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(tasks, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            if c <= 1: continue
            ax.text(x, y, "{0:.1f}".format(c), ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=len(target), bbox_to_anchor=(0, 0),
              loc='upper left', fontsize='small')
    
    plt.subplots_adjust(left=0.05, right=0.99)
    plt.title('{0}: (User-{1:02d}, UI-{2})'.format(TITLE[attr], userId, UI[uiId]))
    # plt.savefig(PIC.format(attr, "barh", userId, uiId))
    plt.show()
    plt.close()

def barh_plot_all(userId, attr, target=OBJ_LABEL, figsize=FIGSIZE, save=SAVE, pdf=None):
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=figsize)

    for uiId, ax in enumerate(axs):
        data = all[userId, uiId]
        tasks = [ "{0}-{1}*".format(i, ROBOT_NUM[d["state"]])  if d["collision_flag"] else "{0}-{1} ".format(i, ROBOT_NUM[d["state"]]) for i, d in enumerate(data)]
        data = [ dict( [(label, count) for label, count in zip(*np.unique(sub[attr], return_counts=True))]) for sub in data ]
        data = [ [dic.get(label, 0) for label in target] for dic in data]
        data = np.array([ l/sum(l)*100 for l in data])
        data_cum = data.cumsum(axis=1)
        category_colors = plt.get_cmap('RdYlGn')(
            np.linspace(0.15, 0.85, data.shape[1]))
        
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.xaxis.set_visible(False)  # hide the x-axis
        ax.set_xlim(0, data.sum(axis=1).max())  # scale x-axis
        ax.set_ylim(-1, len(tasks) -0.5)

        for i, (colname, color) in enumerate(zip(target, category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths
            ax.barh(tasks, widths, left=starts, height=0.5,
                    label=colname, color=color)
            xcenters = starts + widths / 2

            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            for y, (x, c) in enumerate(zip(xcenters, widths)):
                if c <= 1: continue
                ax.text(x, y, "{0:.1f}".format(c), ha='center', va='center',
                        color=text_color)
        ax.text(-0.08, 0.5, UI[uiId], transform=ax.transAxes, fontsize=10, va='center')
    
    plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.05, hspace=0.03)
    fig.suptitle('{0}: (User-{1:02d})'.format(TITLE[attr], userId))
    fig.legend(target, ncol=len(target), bbox_to_anchor=(0.5, 0), loc='lower center', fontsize='small')

    if save:
        plt.savefig(PATH.format(attr+"_barh", userId, 9, TYPE), dpi=DPI)
    elif pdf:
        pdf.savefig(fig)
    else:
        plt.show()
    plt.close()
    

def robot_distance(userId, uiId, figsize=FIGSIZE, save=SAVE, pdf=None):
    subtaskData = all[userId, uiId]
    plot_data = []
    for i, sub in enumerate(subtaskData):
        min_dist = []
        for j in range(len(sub["pos"])):
            mdist = 10**10
            for k in sub["robot"]:
                if k["r_id"][j] == 99: continue
                mdist = min(distance(sub["pos"][j], k["pos"][j]), mdist)
            min_dist.append(mdist)
        # print("({0}, {1}) distance: {2}".format(sub["userId"], sub["uiId"], sum(min_dist)/len(min_dist)))
        plot_data.append(min_dist)
    time_series_plot("Robot Distance", plot_data, subtaskData, ylim=(0, 6), figsize=figsize, save=save, pdf=pdf)

def rot_y(userId, uiId, figsize=FIGSIZE, save=SAVE, pdf=None):
    subtaskData = all[userId, uiId]
    time_series_plot("Rotation_y", [data["rot"][:, 1] for data in subtaskData], subtaskData, figsize=figsize, save=save, pdf=pdf)

def rot_y_diff(userId, uiId, figsize=FIGSIZE, save=SAVE, pdf=None):
    subtaskData = all[userId, uiId]
    plot_data = [ np.diff(data["rot"], axis=0, prepend=[data["rot"][0]])[:, 1] for data in subtaskData ]
    time_series_plot("Rotation_y Diff [1 frame]", plot_data, subtaskData, ylim=(-5.5, 5.5), figsize=figsize, save=save, pdf=pdf)

def rot_y_diff_n(userId, uiId, n:int, figsize=FIGSIZE, save: str =None, pdf=None):
    subtaskData = all[userId, uiId]
    plot_data = [ np.concatenate([np.full(n, 0), (data["rot"][n:] - data["rot"][:-n])[:, 1]]) for data in subtaskData ]
    time_series_plot(f"Rotation_y Diff [{n} frame]", plot_data, subtaskData, figsize=figsize, save=save, pdf=pdf)

def bpm(userId, uiId, figsize=FIGSIZE, save:str =None, pdf=None):
    subtaskData = all[userId, uiId]
    time_series_plot("Heart Rate", [data["bpm"] for data in subtaskData], subtaskData, figsize=figsize, save=save, pdf=pdf)

def time_series_plot(name:str, plot_data, subtaskData, ylim: tuple[float, float]=None, xlim:tuple[float, float]=None, figsize:tuple[float, float]=FIGSIZE, save: str=None, pdf=None):
    fig, axs = plt.subplots(nrows=9, ncols=1, figsize=figsize)
    fig.suptitle("{0}: User-{1:02d}, UI-{2}".format(name, subtaskData[0]["userId"], UI[subtaskData[0]["uiId"]]))
    fig.subplots_adjust(hspace=0.3, left=0.05, right=0.99)
    
    x = [ data["time"] - data["time"][0] for data in subtaskData ]

    if not xlim:
        xlim = (0, max([r[-1] for r in x]))
        xmax = xlim[1]
    else:
        xmax = xlim[1]

    if not ylim:
        ymin, ymax = min([min(r) for r in plot_data]), max([max(r) for r in plot_data])
        dy = (ymax - ymin) / 10
        ylim = ymin-dy, ymax+dy
    
    for i, pd in enumerate(plot_data):
        axs[i].set_ylim(*ylim)
        axs[i].set_xlim(*xlim)
        axs[i].text(-0.08, 0.5, subtaskData[i]["label"], transform=axs[i].transAxes, fontsize=10, va='center')
        axs[i].plot(x[i], pd)
        obj_bgcolor(axs[i], x[i], subtaskData[i]["obj"])
        collision_plot(axs[i], x[i], subtaskData[i]["collision"], xmax)
        warning_bgcolor(axs[i], x[i], subtaskData[i]["warning"])

    plt.tight_layout()
    if save:
        plt.savefig(PATH.format(save, subtaskData[0]["userId"], subtaskData[1]["uiId"], TYPE), dpi=DPI)
    elif pdf:
        pdf.savefig(fig)
    else:
        plt.show()
    plt.close()

def obj_bgcolor(ax, x, obj_list):
    prev = obj_list[0]
    pt = x[0]
    for t, obj in zip(x, obj_list):
        if obj != prev:
            if prev == "Robot" or prev == "Arrow":
                ax.axvspan(pt, t, facecolor=OBJ_COLOR[prev], alpha=0.5)
            prev = obj
            pt = t
    if prev == "Robot" or prev == "Arrow":
        ax.axvspan(pt, x[-1], facecolor=OBJ_COLOR[prev], alpha=0.5)

def warning_bgcolor(ax, x, warning_list):
    prev = warning_list[0]
    pt = x[0]
    for t, obj in zip(x, warning_list):
        if obj != prev:
            if prev != 0:
                ax.axvspan(pt, t, 0, prev*0.1, facecolor="green", alpha=0.5)
            prev = obj
            pt = t
    if prev != 0:
        ax.axvspan(pt, x[-1], 0, prev*0.1, facecolor="green", alpha=0.5)

def collision_plot(ax, x, collision_list, xmax=None):
    if not xmax:
        xmax = len(collision_list)
    for t, collision in zip(x, collision_list):
        if collision:
            ax.text(t/xmax, 0.99, "*", fontsize=15, color="magenta", ha='center', va='top', transform=ax.transAxes)

def calculate_change(userId, uiId, attr):
    data = load_subtask(userId, uiId)
    for i, sub in enumerate(data):
        changes = []
        sub = sub[attr]
        prev = np.array([0]*len(sub[0]))
        NONE_VEC = np.zeros(sub[0].shape)

        for vec in sub:
            if (vec == NONE_VEC).all(): continue

            if (prev != NONE_VEC).all():
                change = np.linalg.norm(vec - prev)
                changes.append(change)
            prev = vec

        total_change = sum(changes)
        average_change = total_change / len(changes) if changes else 0

        print("({0}, {1}, {2}) Total change: {3}, Average change: {4})".format(userId, uiId, i, total_change, average_change))

def save_pdf(func, userIdRange=range(3, N), uiIdRange=range(0, 4)):
    pdf_path = PPATH.format(func.__name__)
    pdf = PdfPages(pdf_path)
    try: 
        for userId in userIdRange:
            for uiId in uiIdRange:
                func(userId, uiId, pdf=pdf)
    except Exception as e:
        print("Error in {0}: {1}".format(func.__name__, e))
    pdf.close()

def show_plot(func, userIdRange=range(3, N), uiIdRange=range(0, 4)):
    try:
        for userId in userIdRange:
            for uiId in uiIdRange:
                func(userId, uiId)
    except Exception as e:
        print("Error in {0}: {1}".format(func.__name__, e))
        traceback.print_exc()

def save_pic(func, userIdRange=range(3, N), uiIdRange=range(0, 4)):
    for userId in userIdRange:
        for uiId in uiIdRange:
            try:
                func(userId, uiId, save=func.__name__)
            except Exception as e:
                print("Error in {0}: {1}".format(func.__name__, e))
                traceback.print_exc()

if __name__ == "__main__":
    try:
        all = load_all()
        # total, average = calculate_change(i, j, "pos")
        # print(i, j, "Total change: {0}, Average change: {1}".format(total, average))
        # calculate_change(i, j, "qua")
        # pie_plot(i, j, "obj", OBJ_LABEL)
        # barh_plot(i, j, "obj", OBJ_LABEL)
        # robot_distance(3, 0)
        # barh_plot(3, 0, "obj", OBJ_LABEL)
        # barh_plot_all(3, "obj", OBJ_LABEL)
        # rot_y_plot(3, 0)
        # robot_distance(3, 0)
        # rot_y_diff_plot(3, 0)
        # save_pdf(bpm)
        show_plot(lambda x, y: rot_y_diff_n(x, y, 5))
        # rot_y_diff(3, 3)
        # rot_y_diff_n(3, 3, 10)
        pass
    except Exception as e:
        traceback.print_exc()