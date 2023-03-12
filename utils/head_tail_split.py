import numpy as np

ratio_list = [0.1, 0.2]

def split_items(data):
    # we here select 0.2 for model evaluation. 0.1 presents the similar results.


    item_degree_dict = {}

    # Loop through both the train and test dictionaries in the data object
    for item_dict in [data.train_dict.items(), data.test_dict.items()]:
        # Unpack the key-value pairs from the dictionary
        for key, values in item_dict:
            # Loop through each value (user) associated with the current key (item)
            for v in values:
                # If the current value (user) is already in the item_degree_dict, increase its degree by 1
                if v in item_degree_dict.keys():
                    item_degree_dict[v] += 1
                # Otherwise, initialize a new entry with a degree of 0
                else:
                    item_degree_dict[v] = 0

    # Sort the item_degree_dict dictionary by degree, descending
    sorted_item_degree_dict = sorted(item_degree_dict.items(), key=lambda x: x[1], reverse=True)
    # Initialize an empty list to store the items that will be included in each ratio
    items = []

    # Loop through each ratio specified in the ratio_list
    for ratio in ratio_list:
        # Calculate the number of top items based on the current ratio
        top_items = sorted_item_degree_dict[:int(ratio*len(item_degree_dict))]
        # Calculate the number of bottom items based on the current ratio
        bottom_items = sorted_item_degree_dict[int(ratio*len(item_degree_dict)):]
        # Add the current set of top and bottom items to the items list
        items.append([set(np.array(top_items)[:, 0]), set(np.array(bottom_items)[:, 0])])

    return items


def output_format(epoch, ix, recall, ndcg):
    return ('Test:{:3d}\t Split ratio: {:.2f}'
            '\n\tRecall@5\t{:.4f}\t{:.4f}\t{:.4f}'
            '\n\tRecall@10\t{:.4f}\t{:.4f}\t{:.4f}'
            '\n\tRecall@20\t{:.4f}\t{:.4f}\t{:.4f}'
            '\n\tRecall@50\t{:.4f}\t{:.4f}\t{:.4f}'
            '\n\tNDCG@5  \t{:.4f}\t{:.4f}\t{:.4f}'
            '\n\tNDCG@10  \t{:.4f}\t{:.4f}\t{:.4f}'
            '\n\tNDCG@20  \t{:.4f}\t{:.4f}\t{:.4f}'
            '\n\tNDCG@50  \t{:.4f}\t{:.4f}\t{:.4f}\n\n').format(
                epoch + 1, ratio_list[ix],
                recall[ix][0][0], recall[ix][0][1], recall[ix][0][2],
                recall[ix][1][0], recall[ix][1][1], recall[ix][1][2],
                recall[ix][2][0], recall[ix][2][1], recall[ix][2][2],
                recall[ix][3][0], recall[ix][3][1], recall[ix][3][2],
                ndcg[ix][0][0], ndcg[ix][0][1], ndcg[ix][0][2],
                ndcg[ix][1][0], ndcg[ix][1][1], ndcg[ix][1][2],
                ndcg[ix][2][0], ndcg[ix][2][1], ndcg[ix][2][2],
                ndcg[ix][3][0], ndcg[ix][3][1], ndcg[ix][3][2],
    )