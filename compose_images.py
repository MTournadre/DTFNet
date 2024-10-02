import cv2
import glob
import os
import numpy as np

from typing import List

def main(
        sequence_path: os.PathLike,
        ref_idx: int,
        method_names: List[str],
        method_paths: List[os.PathLike],
        output_path: os.PathLike,
        start_idx: int = 0,
        ):

    assert len(method_names) > 0, "Empty input!"
    assert len(method_names) == len(method_paths)
    nb_methods = len(method_names)

    traj_dict = {}
    flow_dict = {}
    vis_dict = {}

    for name, path in zip(method_names,method_paths):
        traj_list = sorted( glob.glob(os.path.join(path,"traj_*.png")) )
        traj_dict[name] = [ cv2.imread( f, cv2.IMREAD_COLOR ) for f in traj_list ]
        flow_list = sorted( glob.glob(os.path.join(path,"flow_*.png")) )
        flow_dict[name] = [ cv2.imread( f, cv2.IMREAD_COLOR ) for f in flow_list ]
        vis_list = sorted( glob.glob(os.path.join(path,"vis_*.png")) )
        if len(vis_list) > 0:
            vis_dict[name] = [ cv2.imread( f, cv2.IMREAD_COLOR ) for f in vis_list ]

    # Sequence
    seq_len = len(traj_list)
    ref_frame = cv2.imread( os.path.join(method_paths[0],"ref.png") )
    H, W, _ = ref_frame.shape
    ext = ["png","jpg","bmp"]
    frames = []
    for e in ext:
        frames += sorted( glob.glob(os.path.join(sequence_path,f"*.{e}")) )
    frames = [ cv2.imread( f, cv2.IMREAD_COLOR ) for f in frames[start_idx:start_idx+seq_len] ]
    H_, W_, _ = frames[0].shape
    if H_ != H or W_ != W:
        frames = [ cv2.resize(f,(W,H)) for f in frames ]

    # Wheel (legend)
    wheel = cv2.imread( os.path.join(method_paths[0],"wheel.png") )

    # Constants
    legend_sz = 320
    wheel_sz = wheel.shape[0]
    border = 15
    fontSize = 24
    width = max( legend_sz + 5*border + 3*W, 1200 )
    width = width + (2-(width%2))
    height = (nb_methods+3)*border + (nb_methods+1)*H + fontSize
    height = height + (2-(height%2)) # must be multiple of 2
    font = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 1.
    text_color = (0,0,0)
    thickness = 2
    line = cv2.LINE_AA

    os.makedirs(output_path,exist_ok=True)

    # Compose
    for i_f in range(seq_len):
        img = 255*np.ones((height,width,3),dtype=np.uint8)
        def putImg(I,i,j):
            # Some methods return larger images (multiple of 8)
            # So we crop the result to the original size
            H_ = min(H,I.shape[0])
            W_ = min(W,I.shape[1])
            img[i:i+H_,j:j+W_] = I[:H,:W]

        # Frame + ref
        putImg(frames[i_f],border,legend_sz+2*border+W//2)
        putImg(ref_frame,border,legend_sz+3*border+W//2+W)

        for i_m, name in enumerate(method_names):
            # Trajectory
            putImg(traj_dict[name][i_f],(i_m+2)*border+(i_m+1)*H,legend_sz+2*border)

            # Optical-flow
            putImg(flow_dict[name][i_f],(i_m+2)*border+(i_m+1)*H,legend_sz+3*border+W)

            # Visibility
            if name in vis_dict:
                putImg(vis_dict[name][i_f],(i_m+2)*border+(i_m+1)*H,legend_sz+4*border+2*W)

            # Name
            cv2.putText(img,name,(border,H//2+(i_m+2)*border+(i_m+1)*H+fontSize),
                        font,fontScale,text_color,thickness,line)

        # Legend
        putImg(wheel,2*border+H-wheel_sz,legend_sz+2*border+border//2+W+W//2-wheel_sz//2)

        cv2.putText(img,f"Frame {i_f+start_idx}",(legend_sz+2*border+W//2-200,border+H//2+fontSize),
                    font,fontScale,text_color,thickness,line)
        cv2.putText(img,"Reference",(legend_sz+4*border+2*W+W//2,border+H//2+fontSize//2),
                    font,fontScale,text_color,thickness,line)
        cv2.putText(img,f"(Frame {ref_idx})",(legend_sz+4*border+2*W+W//2,border+H//2+fontSize//2+10+fontSize),
                    font,fontScale,text_color,thickness,line)

        cv2.putText(img,"Trajectory",(legend_sz+border+W//2-60,(nb_methods+2)*border+(nb_methods+1)*H+fontSize),
                    font,fontScale,text_color,thickness,line)
        cv2.putText(img,"Optical-Flow",(legend_sz+2*border+W//2+W-80,(nb_methods+2)*border+(nb_methods+1)*H+fontSize),
                    font,fontScale,text_color,thickness,line)
        cv2.putText(img,"Visibility",(legend_sz+3*border+W//2+2*W-50,(nb_methods+2)*border+(nb_methods+1)*H+fontSize),
                    font,fontScale,text_color,thickness,line)

        # Save resulting image
        out_img_path = os.path.join(output_path,f"frame_{i_f:04d}.png")
        cv2.imwrite( out_img_path, img )
        print( f"  Saved {out_img_path}", end='\r' )

    print("")

if __name__ == "__main__":
    import argparse
    from dtf_core.parsing import PathType

    argparser = argparse.ArgumentParser( prog="compose_images",
            description="Compose images from all models on the same sequence",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter )

    argparser.add_argument( "--sequence_path", type=PathType(exists=True,type='dir'),
                           help="Original sequence" )
    argparser.add_argument( "--ref_idx", type=int,
                           help="Index to reference frame",
                           default=0 )
    argparser.add_argument( "--method_names", type=str,
                           nargs="+",
                           help="Name of each method." )
    argparser.add_argument( "--method_paths", type=PathType(exists=True,type='dir'),
                           nargs="+",
                           help="Output path of each method." )
    argparser.add_argument( "--output_path", type=PathType(exists=None,type='dir'),
                           help="Path to output directory" )
    argparser.add_argument( "--start_idx", type=int,
                           help="Index to reference frame",
                           default=0 )

    args = argparser.parse_args()
    main(**vars(args))
