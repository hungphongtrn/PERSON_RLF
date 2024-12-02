from model import objectives as loss
import torch


def prepare_features(dim=512, batch_size=10):
    image_feat = torch.randn(batch_size, dim)
    text_feat = torch.randn(batch_size, dim)
    return image_feat.to("cuda"), text_feat.to("cuda")


def prepare_labels(batch_size=10, use_sigmoid=False):
    pids = torch.randint(0, 10, (batch_size,))
    sim_targets = torch.eq(pids.view(-1, 1), pids.view(1, -1)).float()
    if use_sigmoid:
        sim_targets = (
            -torch.ones_like(sim_targets) + 2 * sim_targets
        )  # -1 if different, 1 if same
        return sim_targets

    return sim_targets / sim_targets.sum(
        dim=1, keepdim=True
    )  # normalize the true matching distribution


def test_loss():
    image_feat, text_feat = prepare_features()
    batch_size = image_feat.size(0)
    embedding_dim = image_feat.size(1)

    logit_scale = 5.0
    logit_bias = 2.0

    # SDM loss
    sim_targets = prepare_labels(use_sigmoid=False).to("cuda")
    sdm_loss_sigmoid = loss.compute_sdm(
        image_features=image_feat,
        text_features=text_feat,
        sim_targets=sim_targets,
        logit_scale=logit_scale,
        logit_bias=logit_bias,
        use_sigmoid=True,
    )
    sdm_loss = loss.compute_sdm(
        image_features=image_feat,
        text_features=text_feat,
        sim_targets=sim_targets,
        logit_scale=logit_scale,
        logit_bias=logit_bias,
        use_sigmoid=False,
    )

    # ID loss
    sim_targets = prepare_labels(use_sigmoid=False).to("cuda")
    classifier = torch.nn.Linear(embedding_dim, batch_size).to("cuda")
    image_logits = classifier(image_feat)
    text_logits = classifier(text_feat)
    id_loss = loss.compute_id(
        image_logits=image_logits, text_logits=text_logits, labels=sim_targets
    )

    # CITC loss
    citc_loss = loss.compute_citc(
        image_features=image_feat,
        text_features=text_feat,
        logit_scale=logit_scale,
        logit_bias=logit_bias,
        inmodal_weight=0.25,
        intermodal_weight=0.25,
    )

    # RITC loss
    sim_targets = prepare_labels(use_sigmoid=False).to("cuda")
    sigmoid_ritc_loss = loss.compute_ritc(
        image_features=image_feat,
        text_features=text_feat,
        logit_scale=logit_scale,
        logit_bias=logit_bias,
        sim_targets=sim_targets,
        use_sigmoid=True,
    )
    ritc_loss = loss.compute_ritc(
        image_features=image_feat,
        text_features=text_feat,
        logit_scale=logit_scale,
        logit_bias=logit_bias,
        sim_targets=sim_targets,
        use_sigmoid=False,
    )

    # Contrastive loss
    image_features_stopped = image_feat.clone().detach()
    text_features_stopped = text_feat.clone().detach()
    # Use sigmoid
    sim_targets = prepare_labels(use_sigmoid=True).to("cuda")
    sigmoid_contrastive_loss = loss.compute_constrative(
        image_features=image_feat,
        text_features=text_feat,
        image_features_stopped=image_features_stopped,
        text_features_stopped=text_features_stopped,
        sim_targets=sim_targets,
        alpha=0.5,
        logit_scale=logit_scale,
        logit_bias=logit_bias,
        use_sigmoid=True,
    )
    # Do not use sigmoid
    sim_targets = prepare_labels(use_sigmoid=False).to("cuda")
    contrastive_loss = loss.compute_constrative(
        image_features=image_feat,
        text_features=text_feat,
        image_features_stopped=image_features_stopped,
        text_features_stopped=text_features_stopped,
        sim_targets=sim_targets,
        alpha=0.5,
        logit_scale=logit_scale,
        logit_bias=logit_bias,
        use_sigmoid=False,
    )
    print(
        f"SDM Loss (Sigmoid): {sdm_loss_sigmoid.item()}\n",
        f"SDM Loss: {sdm_loss.item()}\n",
        f"ID Loss: {id_loss.item()}\n",
        f"CITC Loss: {citc_loss.item()}\n",
        f"RITC Loss (Sigmoid): {sigmoid_ritc_loss.item()}\n",
        f"RITC Loss: {ritc_loss.item()}\n",
        f"Contrastive Loss (Sigmoid): {sigmoid_contrastive_loss.item()}\n",
        f"Contrastive Loss: {contrastive_loss.item()}\n",
    )


if __name__ == "__main__":
    test_loss()
