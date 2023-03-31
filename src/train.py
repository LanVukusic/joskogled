format_single = lambda key, value : "{}: {:.2f}| ".format(key, value)
format_compute = lambda key, value : "  -: {}: {:.2f}\n".format(key, value)

def single_metrics(pred, labels, metrics):
    value = ""
    for metric_name, metric in metrics.items():
        value += format_single(metric_name, metric(out, years_to_cancer))
    return value


def compute_metrics(pred, labels, metrics):
    value = ""
    for metric_name, metric in list(list(metrics.items())):
        value += format_compute(metric_name, metric.compute())
    [metric.reset() for metric in metrics.values()] # reset metrics
    return value

def train(dtl_train, dtl_val, trainer, epochs, metrics):
# train for number of epochs
    for epoch in range(epochs):
        loss_values = []

        # model.train()
        for batch_idx, (
            patient_id,
            l_cc,
            l_mlo,
            r_cc,
            r_mlo,
            years_to_cancer,
        ) in enumerate(dtl_train):
            loss, out = trainer.train((l_cc, l_mlo, r_cc, r_mlo), years_to_cancer)
            loss_values.append(loss.item())

            # calculate and print metrics for one batch
            metric_values = single_metrics(out, years_to_cancer, metrics)
            print(metric_values + "| " + loss.item(), flush=True)

        # print metrics and compute
        print()
        print("EPOCH {}:".format(epoch))
        print(
            compute_metrics(out, years_to_cancer, metrics) +
            format_compute(loss_values.mean())
        )
        print()

        # after every epoch calculate metrics and loss on val data
        for batch_idx, (
            patient_id,
            l_cc,
            l_mlo,
            r_cc,
            r_mlo,
            years_to_cancer,
        ) in enumerate(dtl_val):
            loss, out = trainer.eval((l_cc, l_mlo, r_cc, r_mlo), years_to_cancer)
            loss_values.append(loss)
            # add data to metrics
            single_metrics(out, years_to_cancer, metrics)

        print("\nVALIDATION: ")
        print(
            compute_metrics(out, years_to_cancer, metrics) +
            format_compute(loss_values.mean())
        )
