
def recall_at_k_per_user(actual, pred, k):
    act_set = set(actual)
    return len(act_set & set(pred[:k])) / float(len(act_set))


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(actual)
    true_users = 0
    for i, v in actual.items():
        act_set = set(v)
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    assert num_users == true_users
    return sum_recall / true_users

def recall_at_k_head_tail(actual, predicted, topk, top_bottom_item):

    sum_recall_top = 0.0
    sum_recall_bottom = 0.0
    sum_recall = 0.0

    # Get the number of users
    num_users = len(actual)
    true_users = 0

    # Split the items into top and bottom groups
    top_items, bottom_items = top_bottom_item

    # Loop through each user's action
    for i, v in actual.items():

        # Get the sets of actual and predicted items
        act_set = set(v)
        pred_set = set(predicted[i][:topk])

        # Compute recall values only for non-empty actual sets
        if len(act_set) != 0:
            # Compute recall of all items for the current user
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            # Compute recall where predicted item is not among bottom items for the current user
            sum_recall_top += len(act_set & pred_set - bottom_items) / float(len(act_set))
            # Compute recall where predicted item is not among top items for the current user
            sum_recall_bottom += len(act_set & pred_set - top_items) / float(len(act_set))
            # Increment the number of true users for whom we calculated recall values
            true_users += 1

    assert num_users == true_users

    # Return the average recall values (total recall divided by the number of users)
    return [sum_recall / true_users, sum_recall_top/true_users, sum_recall_bottom/true_users]
