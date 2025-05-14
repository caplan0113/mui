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
    fig, axs = plt.subplots(nrows=9, ncols=1, figsize=figsize)
    fig.suptitle("Robot distance: User-{0:02d}, UI-{1}".format(subtaskData[0]["userId"], UI[subtaskData[0]["uiId"]]))
    fig.subplots_adjust(hspace=0.3, left=0.05, right=0.99)
    plot_data = []
    # tasks = [ "{0}*".format(ROBOT_NUM[d["state"]])  if d["collision_flag"] else "{0} ".format(ROBOT_NUM[d["state"]]) for i, d in enumerate(subtaskData)]
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

    xmax = max([len(r) for r in plot_data])
    ymin, ymax = min([min(r) for r in plot_data]), max([max(r) for r in plot_data])
    for i, min_dist in enumerate(plot_data):
        axs[i].set_ylim(0, ymax*1.1)
        axs[i].set_xlim(0, xmax)
        # axs[i].set_xticks(range(0, len(min_dist), 100))
        # axs[i].set_xticklabels(range(0, len(min_dist), 100))
        axs[i].text(-0.08, 0.5, subtaskData[i]["label"], transform=axs[i].transAxes, fontsize=10, va='center')
        axs[i].plot(min_dist)
        obj_bgcolor(axs[i], subtaskData[i]["obj"])
        collision_plot(axs[i], subtaskData[i]["collision"], xmax)
        warning_bgcolor(axs[i], subtaskData[i]["warning"])

    plt.tight_layout()
    
    if save:
        plt.savefig(PATH.format("robot_distance", userId, uiId, TYPE), dpi=DPI)
    elif pdf:
        pdf.savefig(fig)
    else:
        plt.show()
    
    plt.close()

def rot_y_plot(userId, uiId, figsize=FIGSIZE, save=SAVE, pdf=None):
    subtaskData = all[userId, uiId]
    fig, axs = plt.subplots(nrows=9, ncols=1, figsize=figsize)
    fig.suptitle("Rotation_y: User-{0:02d}, UI-{1}".format(subtaskData[0]["userId"], UI[subtaskData[0]["uiId"]]))
    fig.subplots_adjust(hspace=0.3, left=0.05, right=0.99)
    # tasks = [ "{0}*".format(ROBOT_NUM[d["state"]])  if d["collision_flag"] else "{0} ".format(ROBOT_NUM[d["state"]]) for i, d in enumerate(subtaskData)]

    xmax = max([len(r["rot"]) for r in subtaskData])
    for i, data in enumerate(subtaskData):
        # axs[i].set_ylim(ymin*1.1, ymax*1.1)
        axs[i].set_xlim(0, xmax)
        # axs[i].set_xticks(range(0, len(min_dist), 100))
        # axs[i].set_xticklabels(range(0, len(min_dist), 100))
        axs[i].text(-0.08, 0.5, data["label"], transform=axs[i].transAxes, fontsize=10, va='center')
        axs[i].plot(data["rot"][:, 1])
        obj_bgcolor(axs[i], data["obj"])
        collision_plot(axs[i], data["collision"], xmax)
        warning_bgcolor(axs[i], data["warning"])

    plt.tight_layout()
    # plt.show()
    if save:
        plt.savefig(PATH.format("rot_y", userId, uiId, TYPE), dpi=DPI)
    elif pdf:
        pdf.savefig(fig)
    else:
        plt.show()
    plt.close()


def rot_y_diff_plot(userId, uiId, figsize=FIGSIZE, save=SAVE, pdf=None):
    subtaskData = all[userId, uiId]
    fig, axs = plt.subplots(nrows=9, ncols=1, figsize=figsize)
    fig.suptitle("Rotation_y Diff: User-{0:02d}, UI-{1}".format(subtaskData[0]["userId"], UI[subtaskData[0]["uiId"]]))
    fig.subplots_adjust(hspace=0.3, left=0.05, right=0.99)
    # tasks = [ "{0}*".format(ROBOT_NUM[d["state"]])  if d["collision_flag"] else "{0} ".format(ROBOT_NUM[d["state"]]) for i, d in enumerate(subtaskData)]

    xmax = max([len(r["rot"])-1 for r in subtaskData])
    for i, data in enumerate(subtaskData):
        # axs[i].set_ylim(ymin*1.1, ymax*1.1)
        axs[i].set_xlim(0, xmax)
        # axs[i].set_xticks(range(0, len(min_dist), 100))
        # axs[i].set_xticklabels(range(0, len(min_dist), 100))
        axs[i].text(-0.08, 0.5, data["label"], transform=axs[i].transAxes, fontsize=10, va='center')
        axs[i].plot(np.diff(data["rot"], axis=0)[:, 1])
        warning_bgcolor(axs[i], data["warning"])
        obj_bgcolor(axs[i], data["obj"])
        collision_plot(axs[i], data["collision"], xmax)

    plt.tight_layout()
    if save:
        plt.savefig(PATH.format("rot_y_diff", userId, uiId, TYPE), dpi=DPI)
    elif pdf:
        pdf.savefig(fig)
    else:
        plt.show()
    plt.close()

def bpm_plot(userId, uiId, figsize=FIGSIZE, save=SAVE, pdf=None):
    subtaskData = all[userId, uiId]
    fig, axs = plt.subplots(nrows=9, ncols=1, figsize=figsize)
    fig.suptitle("BPM: User-{0:02d}, UI-{1}".format(subtaskData[0]["userId"], UI[subtaskData[0]["uiId"]]))
    fig.subplots_adjust(hspace=0.3, left=0.05, right=0.99)
    # tasks = [ "{0}*".format(ROBOT_NUM[d["state"]])  if d["collision_flag"] else "{0} ".format(ROBOT_NUM[d["state"]]) for i, d in enumerate(subtaskData)]

    xmax = max([len(r["bpm"]) for r in subtaskData])
    ymin, ymax = min([min(r["bpm"]) for r in subtaskData]), max([max(r["bpm"]) for r in subtaskData])
    dy = (ymax - ymin) / 10
    for i, data in enumerate(subtaskData):
        axs[i].set_ylim(ymin-dy, ymax+dy)
        axs[i].set_xlim(0, xmax)
        # axs[i].set_xticks(range(0, len(min_dist), 100))
        # axs[i].set_xticklabels(range(0, len(min_dist), 100))
        axs[i].text(-0.08, 0.5, data["label"], transform=axs[i].transAxes, fontsize=10, va='center')
        axs[i].plot(data["bpm"])
        obj_bgcolor(axs[i], subtaskData[i]["obj"])
        collision_plot(axs[i], data["collision"], xmax)
        warning_bgcolor(axs[i], data["warning"])

    plt.tight_layout()
    if save:
        plt.savefig(PATH.format("bpm", userId, uiId, TYPE), dpi=DPI)
    elif pdf:
        pdf.savefig(fig)
    else:
        plt.show()
    plt.close()

def obj_bgcolor(ax, obj_list):
    prev = obj_list[0]
    prev_idx = 0
    for i, obj in enumerate(obj_list):
        if obj != prev:
            if prev == "Robot" or prev == "Arrow":
                ax.axvspan(prev_idx, i-1, facecolor=OBJ_COLOR[prev], alpha=0.5)
            prev = obj
            prev_idx = i
    if prev == "Robot" or prev == "Arrow":
        ax.axvspan(prev_idx, len(obj_list)-1, facecolor=OBJ_COLOR[prev], alpha=0.5)

def warning_bgcolor(ax, warning_list):
    prev = warning_list[0]
    prev_idx = 0
    for i, obj in enumerate(warning_list):
        if obj != prev:
            if prev != 0:
                ax.axvspan(prev_idx, i-1, 0, prev*0.1, facecolor="green", alpha=0.5)
            prev = obj
            prev_idx = i
    if prev != 0:
        ax.axvspan(prev_idx, len(warning_list)-1, facecolor="red", alpha=0.5)

def collision_plot(ax, collision_list, xmax=None):
    if not xmax:
        xmax = len(collision_list)
    for i, collision in enumerate(collision_list):
        if collision:
            ax.text(i/xmax, 0.99, "*", fontsize=15, color="magenta", ha='center', va='top', transform=ax.transAxes)

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
                func(userId, uiId, save=False, pdf=pdf)
    except Exception as e:
        print("Error in {0}: {1}".format(func.__name__, e))
    pdf.close()

def show_plot(func, userIdRange=range(3, N), uiIdRange=range(0, 4)):
    try:
        for userId in userIdRange:
            for uiId in uiIdRange:
                func(userId, uiId, save=False, pdf=None)
    except Exception as e:
        print("Error in {0}: {1}".format(func.__name__, e))

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
        # save_pdf(bpm_plot)
        show_plot(bpm_plot)
        pass
    except Exception as e:
        traceback.print_exc()