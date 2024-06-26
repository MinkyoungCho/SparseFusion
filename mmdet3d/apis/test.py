import mmcv
import torch
# from tools.visualize_results import visualize_bbox

def single_gpu_test(model, data_loader, show=False, out_dir=None):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if show:
            model.module.show_results(data, result, out_dir)

        results.extend(result)
        
        
        # if True:
        #     visualize_bbox(
        #         data=data,
        #         data_name=f"sample_{i}",
        #         outputs=result[0]["pts_bbox"], # okay
        #         cfg=cfg,
        #         topk_ncscore_dict=topk_ncscore_dict, 
        #         topk_p_val_dict=topk_p_val_dict,
        #         topk_top3_values_dict=topk_top3_values_dict,
        #         topk_top3_indices_dict=topk_top3_indices_dict,
        #     )
        

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results
