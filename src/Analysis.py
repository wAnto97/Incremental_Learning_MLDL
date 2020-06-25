from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class Analysis():
    def __init__(self):
        pass

    def useful_plots(self,confusion_matrices):
        n_sample = 1000
        classes_for_batch = 10
        previous_accuracies = [None]
        total_accuracies = []
        new_accuracies = []
        batches_accuracies = []
        gaps_old_classes = [None]
        gaps_new_classes = [None]
        gaps = []
        x_gaps = []
        n_classes = []

        for step in range(len(confusion_matrices)):
            count = 0
            sum_true_previous = 0
            sum_true_new = 0
            sum_batch = 0
            tot = n_sample *(step)
            current_matrix = confusion_matrices[step]
            batches_accuracies.append([])
            for el in range(step):
                batches_accuracies[step].append(None)

            for i in range(len(current_matrix)): 
                if step == 0:
                    sum_true_new += current_matrix[i][i]
                    sum_batch += current_matrix[i][i]
                    count += 1

                if step > 0 and i < (len(current_matrix)-10):
                    sum_true_previous += current_matrix[i][i]
                    sum_batch += current_matrix[i][i]
                    count += 1

                if step > 0 and i >= (len(current_matrix)-10):
                    sum_true_new += current_matrix[i][i]
                    sum_batch += current_matrix[i][i]
                    count +=1

                if ((i+1) % classes_for_batch) == 0:
                    b = int(((i+1) / classes_for_batch)-1)
                    batches_accuracies[b].append((100*sum_batch)/n_sample)
                    sum_batch = 0


            new_accuracies.append(100*(sum_true_new/n_sample))
            total_accuracies.append(100*((sum_true_previous + sum_true_new)/(tot + n_sample)))
            n_classes.append(10*(step+1))
            
            if step > 0:
                previous_accuracies.append(100*(sum_true_previous/tot))
                #gaps_old_classes.append(total_accuracies[step - 1] - previous_accuracies[step])
                gaps_old_classes.append(100*(previous_accuracies[step] / total_accuracies[step - 1]))
                gaps.append(previous_accuracies[step])
                gaps.append(total_accuracies[step])
                x_gaps.append(10*(step+1))
                x_gaps.append(10*(step+1))
            else:
                gaps.append(total_accuracies[step])
                x_gaps.append(10*(step+1))

        for i in range(1, len(new_accuracies)):
            gaps_new_classes.append(100*(new_accuracies[i] / 100))

        fig, ax = plt.subplots(figsize=(6,5))

        ax.plot(n_classes, previous_accuracies, marker ='o', label='Accuracy old classes')
        ax.plot(n_classes, new_accuracies, marker ='o', label='Accuracy new classes')
        ax.plot(n_classes, total_accuracies, marker ='o', label='Total accuracy')

        ax.legend()
        plt.xlabel('N Classes')
        plt.ylabel('Accuracy (%)')
        plt.yticks(np.arange(0, 100, 10))
        plt.xticks(np.arange(0, 100, 20))

        plt.tight_layout()
        plt.grid()
        plt.show()

        fig, ax = plt.subplots(figsize=(6,5))

        ax.plot(n_classes, gaps_old_classes, marker ='o', label='Gap old classes')
        #ax.plot(n_classes, gaps_new_classes, marker ='o', label='Accuracy new classes')

        ax.legend()
        plt.xlabel('N Classes')
        plt.ylabel('Gap (%)')
        #plt.yticks(np.arange(0, 11, 1))
        plt.xticks(np.arange(0, 100, 20))

        plt.tight_layout()
        plt.grid()
        plt.show()

        fig, ax = plt.subplots(figsize=(6,5))

        ax.plot(x_gaps, gaps, marker ='o', label='Gap')

        ax.legend()
        plt.xlabel('N Classes')
        plt.ylabel('Gap (%)')
        #plt.yticks(np.arange(0, 11, 1))
        plt.xticks(np.arange(0, 100, 20))

        plt.tight_layout()
        plt.grid()
        plt.show()


        fig, ax = plt.subplots(figsize=(12,10))

        for i in range(10):#len(batches_accuracies)):
            ax.plot(n_classes, batches_accuracies[i], marker ='o', label = f'Accuracy batch {i}')

        ax.legend()
        plt.xlabel('N Classes')
        plt.ylabel('Accuracy (%)')
        plt.yticks(np.arange(0, 100, 10))
        plt.xticks(np.arange(0, 100, 20))

        plt.tight_layout()
        plt.grid()
        plt.show()

    def plotConfMatrix(self,confusion_matrix,title):
        confusion_matrix = np.array(confusion_matrix)
        confusion_matrix = np.log(np.ones(confusion_matrix.shape) + confusion_matrix)
        confusion_matrix = np.transpose(confusion_matrix)
        
        fig,ax=plt.subplots(figsize=(7,5))
        sns.heatmap(confusion_matrix,cmap='terrain',ax=ax)
        plt.axis('off')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title(title)
        plt.show()

    def plotAccTrendComparison(self,accuracies,labels):
        markers = ['^','o','x','+','D','*','v']
        colors = ['green','darkblue','orange','grey','black','purple','aqua']

        _,ax = plt.subplots(figsize=(12,8))
        for index,(acc,label) in enumerate(zip(accuracies,labels)):
            ax.plot(np.arange(10,110, 10),acc,marker=markers[index],color=colors[index],label=label)


        plt.ylabel('Accuracy')
        plt.xlabel('n_classes')
        plt.title('Top-1 accuracy')
        major_ticks = np.arange(0, 1.1, 0.1)
        minor_ticks = np.arange(0, 1, 0.02)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.set_xticks(np.arange(10,110,10))
        ax.set_xlim(xmin=9,xmax=101)
        plt.legend()
        ax.grid(axis='y')
