from model_meta import ModelMeta

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# fckyea custom metrics bby
metrics = ModelMetrics(device=DEVICE)
metrics.add_metric(
    "auroc", torchmetrics.AUROC(task="multiclass", num_classes=2, average="macro")
)
metrics.add_metric("accuracy", torchmetrics.Accuracy(task="multiclass", num_classes=2))
metrics.add_metric(
    "precision",
    torchmetrics.Precision(task="multiclass", num_classes=2, average="macro"),
)


def train(dtl_train, dtl_val, trainer, epochs):
    # train for number of epochs
    for epoch in range(epochs):
        loss_values = []

        for batch_idx, (
            patient_id,
            l_cc,
            l_mlo,
            r_cc,
            r_mlo,
            years_to_cancer,
        ) in enumerate(dtl_train):
            loss, out = trainer.train((l_cc, l_mlo, r_cc, r_mlo), years_to_cancer)
            # loss_values.append(loss.item())
            # print(loss.item())

            # calculate and print metrics for one batch
            metrics.update(
                out,
                years_to_cancer,
                show=True,
                current_epoch=epoch,
                current_batch=batch_idx,
            )

        # print metrics and compute
        metrics.compute(show=True)
        metrics.reset()

        # loss_values = []

        # after every epoch calculate metrics and loss on val data
        metrics.reset()
        for batch_idx, (
            patient_id,
            l_cc,
            l_mlo,
            r_cc,
            r_mlo,
            years_to_cancer,
        ) in enumerate(dtl_val):
            loss, out = trainer.eval((l_cc, l_mlo, r_cc, r_mlo), years_to_cancer)
            # loss_values.append(loss)
            # add data to metrics
            single_metrics(out, years_to_cancer, metrics)
            metrics.update(
                out,
                years_to_cancer,
                show=False,
            )

        print("\nVALIDATION: ")
        # print metrics and compute
        metrics.compute(show=True)
        metrics.reset()
