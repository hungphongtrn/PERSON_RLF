from sympy import use
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sdm(
    image_fetures,
    text_fetures,
    pid,
    logit_scale,
    image_id=None,
    factor=0.3,
    epsilon=1e-8,
):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    t2i_cosine_theta = text_fetures @ image_fetures.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (
        F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon)
    )
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (
        F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon)
    )

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(
        torch.sum(t2i_loss, dim=1)
    )

    return loss


def compute_mlm(
    scores,
    labels,
    ignore_index=1,
):
    """
    Masked Language Model (MLM) loss

    Args:
        scores: output of the model
        labels: ground truth labels
        ignore_index: index to ignore in the loss computation
    """
    # ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
    # return ce(scores, labels)
    loss = F.cross_entropy(scores, labels, ignore_index=ignore_index, reduction="mean")
    return loss


def compute_itc(
    image_features,
    text_features,
    logit_scale,
):
    """
    Image-text contrastive (ITC) loss, InfoNCE

    Args:
        image_features: image features after pooling
        text_features: text features after pooling
        logit_scale: scaling factor for the logits
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(
        start=0, end=batch_size, dtype=torch.int64, device=image_features.device
    )

    # cosine similarity as logits
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels, reduction="mean")
    loss_t = F.cross_entropy(logits_per_text, labels, reduction="mean")
    loss = (loss_i + loss_t) / 2

    return loss


def compute_id(
    image_logits,
    text_logits,
    labels,
):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535

    Args:
        image_logits: image_features aften passing through a classifier
        text_logits: text_features aften passing through a classifier
        labels: ground truth labels for the classification task
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)

    return loss / 2


def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_proj_text = torch.matmul(image_embeddings, image_embeddings.t())
    text_proj_image = torch.matmul(text_embeddings, image_embeddings.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (
        F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon)
    )
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (
        F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon)
    )

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(
        torch.sum(t2i_loss, dim=1)
    )

    return cmpm_loss


def compute_citc(
    image_features,
    text_features,
    logit_scale,
    inmodal_weight,
    intermodal_weight,
    use_sigmoid,
    logit_bias,
):
    """
    Compute cyclic image-text contrastive loss

    Args:
        image_features: image features after pooling
        text_features: text features after pooling
        logit_scale: scaling factor for the logits
        inmodal_weight: scaling factor for the cyclic loss
        intermodal_weight: scaling factor for the cyclic loss
        use_sigmoid: use sigmoid or softmax for the similarity targets
        logit_bias: bias for the logits if using sigmoid
    """
    sim_i2i = logit_scale * image_features @ image_features.t()
    sim_t2t = logit_scale * text_features @ text_features.t()

    sim_i2t = logit_scale * image_features @ text_features.t()
    sim_t2i = logit_scale * text_features @ image_features.t()

    if use_sigmoid:
        sim_i2i = sim_i2i + logit_bias
        sim_t2t = sim_t2t + logit_bias
        sim_i2t = sim_i2t + logit_bias
        sim_t2i = sim_t2i + logit_bias

    inmodal_cyclic_loss = (sim_i2i - sim_t2t).square().mean() / (
        logit_scale * logit_scale
    )
    intermodal_cyclic_loss = (sim_i2t - sim_t2i).square().mean() / (
        logit_scale * logit_scale
    )

    loss = (
        inmodal_weight * inmodal_cyclic_loss
        + intermodal_weight * intermodal_cyclic_loss
    )
    return loss


def compute_ritc(
    image_features,
    text_features,
    logit_scale,
    sim_targets,
    eps=1e-2,
    use_sigmoid=False,
    logit_bias=0,
):
    """
    Compute the reverse image-text contrastive loss

    Args:
        image_features: image features after pooling
        text_features: text features after pooling
        logit_scale: scaling factor for the logits
    """
    # TODO: Seem incorrect, need to check
    if use_sigmoid:
        target_prob = sim_targets
        logits = logit_scale * image_features @ text_features.t() + logit_bias
        sigmoid_prob = F.sigmoid(logits)

        loss = F.kl_div(sigmoid_prob, target_prob, reduction="batchmean")

    else:
        target_prob = (sim_targets + eps).log()
        sim_i2t = logit_scale * image_features @ text_features.t()
        sim_t2i = logit_scale * text_features @ image_features.t()

        prob_i2t = F.log_softmax(sim_i2t, dim=1)
        prob_t2i = F.log_softmax(sim_t2i, dim=1)

        kl_img = F.kl_div(target_prob, prob_i2t, log_target=True, reduction="batchmean")
        kl_txt = F.kl_div(target_prob, prob_t2i, log_target=True, reduction="batchmean")
        loss = (kl_img + kl_txt) / 2

    return loss


def compute_constrative(
    image_features,
    text_features,
    image_features_stopped,
    text_features_stopped,
    sim_targets,
    alpha,
    logit_scale,
    use_sigmoid,
    logit_bias,
):
    """
    Compute constrative loss for image-text pairs
    with soft labeling mechanism

    Args:
        image_features: image features after pooling
        text_features: text features after pooling
        image_features_stopped: stopped gradients image features
        text_features_stopped: stopped gradients text features
        sim_targets: similarity targets for the image-text pairs
        alpha: scaling factor for the similarity targets
        logit_scale: scaling factor for the logits
        use_sigmoid: use sigmoid or softmax for the similarity targets
        logit_bias: bias for the logits if using sigmoid
    """
    if alpha != 0:
        with torch.no_grad():
            # Soft labels for the similarity targets
            sim_i2t_s = logit_scale * image_features_stopped @ text_features.t()
            sim_t2i_s = logit_scale * text_features_stopped @ image_features.t()
            # Soft + hard labels for the similarity targets
            if use_sigmoid:
                sim_targets = alpha * F.sigmoid(sim_i2t_s) + (1 - alpha) * sim_targets

            else:
                sim_i2t_targets = (
                    alpha * F.softmax(sim_i2t_s, dim=1) + (1 - alpha) * sim_targets
                )
                sim_t2i_targets = (
                    alpha * F.softmax(sim_t2i_s, dim=1) + (1 - alpha) * sim_targets
                )
    # If alpha is 0, use the hard labels
    else:
        sim_i2t_targets = sim_targets
        sim_t2i_targets = sim_targets

    if use_sigmoid:
        # Compute the cosine similarity between the image and text features as the logits
        logits = logit_scale * image_features @ text_features.t() + logit_bias
        # Commpute the negative log-likelihood loss
        loss = -torch.sum(F.logsigmoid(logits * sim_targets)) / image_features.shape[0]
        return loss

    else:
        # Compute the cosine similarity between the image and text features as the logits
        sim_i2t = logit_scale * image_features @ text_features.t()
        sim_t2i = logit_scale * text_features @ image_features.t()
        # Commpute the negative log-likelihood loss
        loss_i2t = -torch.sum(
            F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1
        ).mean()
        loss_t2i = -torch.sum(
            F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1
        ).mean()

        # Loss is the average of the two losses
        loss = (loss_i2t + loss_t2i) / 2
        return loss


def compute_simclr(
    image_features_1,
    image_features_2,
    temperature=0.07,
):
    """
    Contrastive learning loss using SimCLR

    Args:
        image_features: image features after pooling
        text_features: text features after pooling
        temperature: temperature for the softmax
    """
    device = image_features_1.device
    batch_size = image_features_1.shape[0]

    # Create labels for the batch
    labels = torch.arange(start=0, end=batch_size, device=device)

    # Similarity between the first augmented image and the second augmented image
    sim_ab = (image_features_1 @ image_features_2.t()) / temperature
    sim_ba = sim_ab.t()

    mask = torch.where(F.one_hot(labels, batch_size) == 0, 0, float("-inf"))
    # Similarity between the first augmented image and other augmented images in the batch
    sim_aa = (image_features_1 @ image_features_1.t()) / temperature + mask
    sim_bb = (image_features_2 @ image_features_2.t()) / temperature + mask

    # Similarity between all images of the 1st augmented images with all other images
    sim_a = torch.cat((sim_ab, sim_aa), dim=1)
    # Similarity between all images of the 2nd augmented images with all other images
    sim_b = torch.cat((sim_ba, sim_bb), dim=1)

    loss_a = F.cross_entropy(sim_a, labels)
    loss_b = F.cross_entropy(sim_b, labels)

    loss = (loss_a + loss_b) / 2

    return loss
