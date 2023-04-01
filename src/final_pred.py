from torch import no_grad


def make_final_pred(dtl_final, model):
    final = []
    model.eval()
    with no_grad():
        for batch_idx, (
                patient_id,
                l_cc,
                l_mlo,
                r_cc,
                r_mlo
        ) in enumerate(dtl_final):
            out = model.forward(l_cc, l_mlo, r_cc, r_mlo).softmax(dim=1)[:, 1].tolist()
            final += list(zip(patient_id, out))
    model.train()
    return final


def save_pred(final, path):
    with open(path, 'w') as f:
        for pair in final:
            f.write('{},{:.4f}\n'.format(pair[0], pair[1]))
