import matplotlib.pyplot as plt
import numpy as np
from lib import *
from matplotlib.backends.backend_pdf import PdfPages

TITLE = {"obj": "Eye tracking", "pos": "Camera Position", "rot": "Camera Rotation", "bpm": "Heart rate", "trigger": "Button", "state": "State", "subTask": "Subtask", "warning": "Warning", "collision": "Collision"}
UI = ["VA", "VO", "AO", "NO"]
OBJ_LABEL = ["Robot", "Arrow", "TaskPanel", "Shelf", "Building", "None"]
OBJ_COLOR = {"Robot": "red", "Arrow": "blue", "TaskPanel": "yellow", "Shelf": "green", "Building": "grey", "None": "grey"}
PATH = "pic/{0}/{1:02d}_{2:01d}.png"
DPI = 300
SAVE = False
FIGSIZE = (11.69, 8.27)
PPATH = "pdf/{0}.pdf"
TASK_TILE = [ # X, Z
    (23, 24),
    (20, 35),
    (20, 12),
    (13.5, 24),
    (10, 35),
    (10, 12),
    (7, 24),
    (20, 35),
    (20, 12),
    (10, 35),
    (10, 12)
]

TASK_LIM = [ # rot, (xmin, xmax), (zmin, zmax)
    (0, (20 ,24), (21, 28)),
    (1, (17, 24), (32, 36)),
    (2, (17, 24), (12, 16)),
    (3, (13.5, 17.5), (21, 28)),
    (1, (7, 14), (32, 36)),
    (2, (7, 14), (12, 16)),
    (3, (7, 11), (21, 28)),
    (1, (17, 24), (32, 36)),
    (2, (17, 24), (12, 16)),
    (1, (7, 14), (32, 36)),
    (2, (7, 14), (12, 16))
]

TASK_ROBOT = [
    [],
    [[("red", (0, 1), (3/4, 1))], [("red", (0, 1), (0, 1/4))]],
    [[("red", (0, 1), (3/4, 1)), ("magenta", (3/7, 4/7), (0, 3.5/4)), ("magenta", (6/7, 1), (0, 3.5/4))], [("red", (0, 1.5/7), (3/4, 1)), ("magenta", (6/7, 1), (0, 3.5/4))]],
    [[("red", (0, 1), (3/4, 1)), ("magenta", (3/7, 4/7), (0, 3.5/4)), ("magenta", (6/7, 1), (0, 3.5/4)), ("orange", (0, 1), (0.5/4, 1.5/4))], [("red", (0, 1.5/7), (3/4, 1)), ("magenta", (6/7, 1), (0, 3.5/4)), ("orange", (0, 1), (0.5/4, 1.5/4))]]
]

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
        plt.savefig(PATH.format(attr+"_barh", userId, 9), dpi=DPI)
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
                mdist = min(distance(sub["pos"][j], k["pos"][j])[0], mdist)
            min_dist.append(mdist)
        # print("({0}, {1}) distance: {2}".format(sub["userId"], sub["uiId"], sum(min_dist)/len(min_dist)))
        plot_data.append(min_dist)
    time_series_plot("Robot Distance", plot_data, subtaskData, ylim=(0, 6), figsize=figsize, save=save, pdf=pdf)

def rot_y(userId, uiId, figsize=FIGSIZE, save=SAVE, pdf=None):
    subtaskData = all[userId, uiId]
    time_series_plot("Rotation_y", [data["rot"][:, 1] for data in subtaskData], subtaskData, figsize=figsize, save=save, pdf=pdf)

def rot_y_diff(userId, uiId, figsize=FIGSIZE, save=SAVE, pdf=None):
    subtaskData = all[userId, uiId]
    plot_data = [ np.abs(np.diff(data["rot"], axis=0, prepend=[data["rot"][0]]))[:, 1] for data in subtaskData ]
    time_series_plot("Rotation_y Diff [1 frame]", plot_data, subtaskData, ylim=(-2, 6), figsize=figsize, save=save, pdf=pdf)

def rot_y_diff_n(userId, uiId, n:int, figsize=FIGSIZE, save: str =None, pdf=None):
    subtaskData = all[userId, uiId]
    plot_data = [ np.concatenate([np.full(n, 0), np.abs(data["rot"][n:] - data["rot"][:-n])[:, 1]]) for data in subtaskData ]
    time_series_plot(f"Rotation_y Diff [{n} frame]", plot_data, subtaskData, ylim=(-5, 15), figsize=figsize, save=save, pdf=pdf)

def bpm(userId, uiId, figsize=FIGSIZE, save:str =None, pdf=None):
    subtaskData = all[userId, uiId]
    time_series_plot("Heart Rate", [data["bpm"] for data in subtaskData], subtaskData, figsize=figsize, save=save, pdf=pdf)

def pos_x_diff_n(userId, uiId, n:int, figsize=FIGSIZE, save:str =None, pdf=None):
    subtaskData = all[userId, uiId]
    plot_data = [ np.concatenate([np.full(n, 0), (data["pos"][n:] - data["pos"][:-n])[:, 0]]) for data in subtaskData ]
    time_series_plot(f"Position X Diff [{n} frame]", plot_data, subtaskData, ylim=(-0.5, 0.5), figsize=figsize, save=save, pdf=pdf)

def pos_z_diff_n(userId, uiId, n:int, figsize=FIGSIZE, save:str =None, pdf=None):
    subtaskData = all[userId, uiId]
    plot_data = [ np.concatenate([np.full(n, 0), (data["pos"][n:] - data["pos"][:-n])[:, 2]]) for data in subtaskData ]
    time_series_plot(f"Position Z Diff [{n} frame]", plot_data, subtaskData, ylim=(-0.5, 0.5), figsize=figsize, save=save, pdf=pdf)

def pos_xz_diff_n(userId, uiId, n:int, figsize=FIGSIZE, save:str =None, pdf=None):
    subtaskData = all[userId, uiId]
    plot_data = [ np.concatenate([np.full(n, 0), distance(data["pos"][n:], data["pos"][:-n])]) for data in subtaskData ]
    time_series_plot(f"Position XZ Diff [{n} frame]", plot_data, subtaskData, ylim=(-0.1, 0.3), figsize=figsize, save=save, pdf=pdf)

def twin_xz_roty_diff_n(userId, uiId, n:int, figsize=FIGSIZE, save:str =None, pdf=None, legends: list[str]=None):
    subtaskData = all[userId, uiId]
    plot_data = [ np.concatenate([np.full(n, 0), distance(data["pos"][n:], data["pos"][:-n])]) for data in subtaskData ]
    plot_data2 = [ np.concatenate([np.full(n, 0), np.abs(data["rot"][n:] - data["rot"][:-n])[:, 1]]) for data in subtaskData ]
    time_series_plot(f"Position XZ Diff & Rotation Y Diff [{n} frame]", plot_data, subtaskData, plot_data2=plot_data2, ylim=(-0.1, 0.3), ylim2=(-5, 15), legends=legends, figsize=figsize, save=save, pdf=pdf)

def time_series_plot(name:str, plot_data, subtaskData, plot_data2=None, legends: list[str]=None, ylim: tuple[float, float]=None, ylim2: tuple[float, float]=None, xlim:tuple[float, float]=None, figsize:tuple[float, float]=FIGSIZE, save: str=None, pdf=None):
    fig, axs = plt.subplots(nrows=9, ncols=1, figsize=figsize)
    fig.suptitle("{0}: User-{1:02d}, UI-{2}".format(name, subtaskData[0]["userId"], UI[subtaskData[0]["uiId"]]))
    fig.subplots_adjust(hspace=0.55, left=0.09, right=0.97, top=0.93, wspace=0, bottom=0.07)


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
    
    lines = []
    for i, pd in enumerate(plot_data):
        axs[i].set_ylim(*ylim)
        # axs[i].set_xlim(*xlim)
        xmax = max(x[i])
        axs[i].set_xlim(0, xmax)
        axs[i].text(-0.09, 0.5, subtaskData[i]["label"], transform=axs[i].transAxes, fontsize=10, va='center')
        line1,  = axs[i].plot(x[i], pd)
        lines.append(line1)
        obj_bgcolor(axs[i], x[i], subtaskData[i]["obj"])
        collision_plot(axs[i], x[i], subtaskData[i]["collision"], xmax)
        warning_bgcolor(axs[i], x[i], subtaskData[i]["warning"])

        if plot_data2:
            ax2 = axs[i].twinx()

            if not ylim2:
                ymin2, ymax2 = min([min(r) for r in plot_data2]), max([max(r) for r in plot_data2])
                dy2 = (ymax2 - ymin2) / 10
                ylim2 = ymin2-dy2, ymax2+dy2
            ax2.set_ylim(*ylim2)
            line2, = ax2.plot(x[i], plot_data2[i], color="orange", alpha=0.7)
            lines.append(line2)

    if legends:
        fig.legend(handles=lines[0:2], labels=legends, ncol=len(legends), bbox_to_anchor=(0.5, 0), loc='lower center', fontsize='small')        

    # plt.tight_layout()
    if save:
        os.makedirs(f"pic/{save}", exist_ok=True)
        plt.savefig(PATH.format(save, subtaskData[0]["userId"], subtaskData[1]["uiId"]), dpi=DPI)
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
                ax.axvspan(pt, t, facecolor=OBJ_COLOR[prev], alpha=0.3)
            prev = obj
            pt = t
    if prev == "Robot" or prev == "Arrow":
        ax.axvspan(pt, x[-1], facecolor=OBJ_COLOR[prev], alpha=0.3)

def warning_bgcolor(ax, x, warning_list):
    prev = warning_list[0]
    pt = x[0]
    for t, obj in zip(x, warning_list):
        if obj != prev:
            if prev != 0:
                ax.axvspan(pt, t, 0, prev*0.15, facecolor="green", alpha=0.5)
            prev = obj
            pt = t
    if prev != 0:
        ax.axvspan(pt, x[-1], 0, prev*0.15, facecolor="green", alpha=0.5)

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

def save_pdf(func, args=dict(), userIdRange=range(3, N), uiIdRange=range(0, 4)):
    pdf_path = PPATH.format(func.__name__)
    pdf = PdfPages(pdf_path)
    try: 
        for userId in userIdRange:
            for uiId in uiIdRange:
                func(userId, uiId, pdf=pdf, **args)
    except Exception as e:
        print("Error in {0}: {1}".format(func.__name__, e))
    pdf.close()
    print("Saved PDF: {0}".format(pdf_path))

def show_plot(func, args=dict(), userIdRange=range(3, N), uiIdRange=range(0, 4)):
    try:
        for userId in userIdRange:
            for uiId in uiIdRange:
                func(userId, uiId, **args)
    except Exception as e:
        print("Error in {0}: {1}".format(func.__name__, e))
        traceback.print_exc()

def save_pic(func, args=dict(), userIdRange=range(3, N), uiIdRange=range(0, 4)):
    for userId in userIdRange:
        for uiId in uiIdRange:
            try:
                func(userId, uiId, save=func.__name__, **args)
            except Exception as e:
                print("Error in {0}: {1}".format(func.__name__, e))
                traceback.print_exc()

def map_func(func, args=dict(), userIdRange=range(3, N), uiIdRange=range(0, 4)):
    for userId in userIdRange:
        for uiId in uiIdRange:
            try:
                func(userId, uiId, **args)
            except Exception as e:
                print("Error in {0}: {1}".format(func.__name__, e))
                traceback.print_exc()

def pos_xz(userId, uiId, figsize=FIGSIZE, save=SAVE, pdf=None):
    subtaskData = all[userId, uiId]

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=figsize)
    axs = axs.flatten()
    fig.subplots_adjust(hspace=0.3, left=0, right=1, top=0.9, wspace=0, bottom=0.05)
    fig.suptitle("Position XZ: User-{0:02d}, UI-{1}".format(userId, UI[uiId]))

    for i, ax in enumerate(axs):
        x, z = subtaskData[i]["pos"][:, 0], subtaskData[i]["pos"][:, 2]
        cx, cz = subtaskData[i]["pos"][:, 0][subtaskData[i]["collision"]], subtaskData[i]["pos"][:, 2][subtaskData[i]["collision"]]
        s, xlim, zlim = TASK_LIM[subtaskData[i]["state"]]
        if s == 0:
            px, py = z, x
            pcx, pcy = cz, cx
            x_lim = zlim; y_lim = xlim
            xlabel = "Z"; ylabel = "X"
        elif s == 1:
            px, py = x, z
            pcx, pcy = cx, cz
            x_lim = xlim; y_lim = zlim
            xlabel = "X"; ylabel = "Z"
        elif s == 2:
            px, py = x, z
            pcx, pcy = cx, cz
            x_lim = xlim; y_lim = zlim
            xlabel = "X"; ylabel = "Z"
        else:
            px, py = z, x
            pcx, pcy = cz, cx
            x_lim = zlim; y_lim = xlim
            xlabel = "Z"; ylabel = "X"
        
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xticks(np.arange(x_lim[0], x_lim[1]+1, 1))
        ax.set_yticks(np.arange(y_lim[0], y_lim[1]+1, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        # ax.set_xlabel(xlabel)
        # ax.set_ylabel(ylabel)
        ax.set_aspect("equal")
        
        if s == 0:
            ax.invert_xaxis()
        elif s == 2:
            ax.invert_xaxis()
            ax.invert_yaxis()
        elif s == 3:
            ax.invert_yaxis()
        
        task_bgcolor(ax, x_lim, s, ROBOT_NUM[subtaskData[i]["state"]], subtaskData[i]["collision_flag"])

        mask0, mask1, mask2, mask3 = subtaskData[i]["warning"]==0, subtaskData[i]["warning"]==1, subtaskData[i]["warning"]==2, subtaskData[i]["warning"]==3
        px0, py0 = px[mask0], py[mask0]
        px1, py1 = px[mask1], py[mask1]
        px2, py2 = px[mask2], py[mask2]
        px3, py3 = px[mask3], py[mask3]
        ax.scatter(px0, py0, color="green", s=1, zorder=1)
        ax.scatter(px1, py1, color="gold", s=1, zorder=1)
        ax.scatter(px2, py2, color="darkorange", s=1, zorder=1)
        ax.scatter(px3, py3, color="red", s=1, zorder=1)
        
        # ax.scatter(px, py, zorder=1, s=1)
        ax.scatter(pcx, pcy, color="red", marker="*", s=50, zorder=2) # collision
        
        ax.set_title("Subtask {0}".format(subtaskData[i]["label"]))

    if save:
        os.makedirs(f"pic/{save}", exist_ok=True)
        plt.savefig(PATH.format(save, subtaskData[0]["userId"], subtaskData[1]["uiId"]), dpi=DPI)
    elif pdf:
        pdf.savefig(fig)
    else:
        plt.show()
    plt.close()

def task_bgcolor(ax, xlim, s, robot_num, collision_flag):
    xmin, xmax = xlim
    ax.axvspan(xmin+3, xmax-3, 3/4, 1, facecolor="green", alpha=0.5)
    robots = TASK_ROBOT[robot_num][not collision_flag]

    for c, x, y in robots:
        x1, x2 = x; y1, y2 = y
        if s % 2 == 0:
            xs = xmax - (xmax - xmin) * x1
            xe = xmax - (xmax - xmin) * x2
        else:
            xs = xmin + (xmax - xmin) * x1
            xe = xmin + (xmax - xmin) * x2
        ax.axvspan(xs, xe, y1, y2, edgecolor=c, fill=False, hatch="/", alpha=0.5, lw=1)

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
        # save_pdf(pos_xz, args=dict())
        # rot_y_diff(3, 3)
        # rot_y_diff_n(3, 3, 10)
        # map_func(pos_xz, args=dict())
        # pos_xz(3, 0)
        # twin_xz_roty_diff_n(3, 0, 5, legends=["Position XZ", "Rotation Y"])
        # save_pdf(pos_xz_diff_n, args=dict(n=5))
        # save_pdf(pos_xz)
        # save_pdf(rot_y_diff_n, args=dict(n=5))
        # save_pdf(bpm)
        # save_pdf(robot_distance)
        # save_pdf(rot_y)
        save_pdf(twin_xz_roty_diff_n, args=dict(n=5, legends=["Position XZ", "Rotation Y"]))
        # pos_xz_diff_n(3, 0, 10)

        pass
    except Exception as e:
        traceback.print_exc()