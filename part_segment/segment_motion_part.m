function segment_motion_part
path = '../pred';
classfile = dir(fullfile(path,'*'));
classname = {classfile.name}';
classname = classname(3:end);
pred_axis_num = 64;
p_number = 2048;
frame_num = 11;
for c_k = 1:size(classname,1)
    temp_class = classname{c_k};
    classpath = [path '/' temp_class];
    file = dir(fullfile(classpath,'*.mat'));
    filename = {file.name}';
    all_class_iou = 0;
    all_class_uvw_err = 0;
    all_class_theta_err = 0;
    all_class_phi_err = 0;
    all_class_xyz_err = 0;
    all_class_type_acc = 0;
    all_class_rebuild_error = 0;
    for i = 1:size(filename,1)
        load([classpath '/' filename{i}])
        % input
        label = reshape(label,p_number,1);
        axis_uvw = reshape(axis_uvw,pred_axis_num,3);
        axis_xyz = reshape(axis_xyz,pred_axis_num,3);
        theta = reshape(theta,pred_axis_num,1)/3.1415629535897932384626*180;
        phi = reshape(phi,pred_axis_num,1);
        theta(theta<10) = 0;
        phi(phi<0.05) = 0;
        abspose_angle = reshape(abspose_angle,pred_axis_num,1)/3.1415926535897*180;
        abspose_trans = reshape(abspose_trans,pred_axis_num,1);
        pp = axis_gt
        axis_gt = reshape(axis_gt,10,9);
        input_trajecotires = reshape(input_gt(:,:,:,1:3),p_number,frame_num,3);
        first_frame = reshape(input_trajecotires(:,1,:),p_number,3);
        last_frame = reshape(input_trajecotires(:,frame_num,:),p_number,3);
        input_trajecotires_cell = mat2cell(input_trajecotires,ones(p_number,1));
        trajectories_dis = cellfun(@(x) compute_trajectories_dis(input_trajecotires,x),input_trajecotires_cell,'Unif',0);
        trajectories_dis = cell2mat(trajectories_dis);
        [sort_a,sort_b] = sort(trajectories_dis,2);
        sort_b_neighbor1 = [sort_b(:,1:1)];
        sort_b_neighbor3 = [sort_b(:,1:3)];
        sort_b_neighbor5 = [sort_b(:,1:5)];
        axis_pred_all = [axis_xyz axis_uvw theta phi];
        axis_pred_all = enhance_axis_by_theta_phi(axis_pred_all);
        no_movement_idx = find(axis_pred_all(:,7)<10&axis_pred_all(:,8)<0.05);
        axis_pred_all(no_movement_idx,:) = [];
        axis_pred_all = [axis_pred_all;0 0 0 0 0 1 0 0];
        num_axis = size(axis_pred_all, 1)
        points_use = first_frame(:,:)
        axis_pred_all_cell = mat2cell(axis_pred_all,ones(size(axis_pred_all,1),1));
        matrix_sequence_pred = cellfun(@(x) compute_matrix_sequence(x, frame_num),axis_pred_all_cell,'Unif',0);
        rebuild_distance = cellfun(@(x) compute_rebuild_distance(x,input_trajecotires,sort_b_neighbor1, frame_num),matrix_sequence_pred,'Unif',0);
        rebuild_distance = cell2mat(rebuild_distance);
        [val,idx] = min(rebuild_distance,[],1);
        statistic = tabulate(idx);
        axis_idx = statistic(:,1);
        count = statistic(:,2);
        
        count_thres = 15;
        while(size(find(count>count_thres),1)<=1)
            if(size(count,1) <= 1)
                break;
            end
            count_thres = count_thres - 5;
        end
        low_ratio_axis = find(count<=count_thres);
        rebuild_distance(low_ratio_axis,:) = 100;

        [val,idx] = min(rebuild_distance,[],1);
        statistic = tabulate(idx);
        axis_idx = statistic(:,1);
        count = statistic(:,2);
        
        loop_axis_idx = axis_idx;
        loop_count = count;
        choose_axis = [];
        choose_axis_idx = [];
        while(true)
            if (~isempty(choose_axis))
                [~,max_idx] = max(loop_count);
                if(loop_count(max_idx)<=count_thres)
                    break;
                end
                if(isempty(max_idx))
                    break;
                end
                max_axis_idx = loop_axis_idx(max_idx);
                belonging_idx = find(idx == max_axis_idx);
                previous_distance = rebuild_distance(choose_axis_idx,belonging_idx);
                now_distance = rebuild_distance(max_axis_idx,belonging_idx);
                now_distance = repmat(now_distance,size(previous_distance,1),1);
                distance_err = abs(previous_distance - now_distance);
                distance_err = min(distance_err,[],1);
                xxx = axis_pred_all(max_axis_idx,:);
                dis_thres = 0;
                if(xxx(7)>10)
                    dis_thres = 0.1;
                elseif(xxx(8)>0)
                    dis_thres = 0.01;
                end
                ratio = sum(distance_err<dis_thres)/size(belonging_idx,2)
                if(ratio>0.8)
                    loop_axis_idx(max_idx) = [];
                    loop_count(max_idx) = [];
                    continue;
                end
            end
            [~,max_idx] = max(loop_count);
            max_axis_idx = loop_axis_idx(max_idx);
            choose_axis_idx = [choose_axis_idx;max_axis_idx];
            choose_axis = [choose_axis;axis_pred_all(max_axis_idx,:)];
            loop_axis_idx(max_idx) = [];
            loop_count(max_idx) = [];
        end
        
        choose_axis_all_cell = mat2cell(choose_axis,ones(size(choose_axis,1),1));
        choose_matrix_sequence_pred = cellfun(@(x) compute_matrix_sequence(x, frame_num),choose_axis_all_cell,'Unif',0);
        choose_rebuild_distance = cellfun(@(x) compute_rebuild_distance(x,input_trajecotires,sort_b_neighbor1,frame_num),choose_matrix_sequence_pred,'Unif',0);
        choose_rebuild_distance = cell2mat(choose_rebuild_distance);
        [choose_val,choose_idx] = min(choose_rebuild_distance,[],1);
        choose_statistic = tabulate(choose_idx);
        choose_axis_idx_1 = choose_statistic(:,1);
        choose_count = choose_statistic(:,2);
        
        all_pred_label = choose_idx(sort_b_neighbor5);
        choose_idx = mode(all_pred_label,2);
        
        GT_seg = generate_seg(label);
        pred_seg = generate_seg(choose_idx);
        GT_seg_cell = mat2cell(GT_seg,ones(size(GT_seg,1),1));
        
        iou_sore = cellfun(@(x) compute_score(x,pred_seg),GT_seg_cell,'Unif',0);
        iou_sore = iou_sore';
        iou_sore = cell2mat(iou_sore);
        [pick_max_score,pick_score_id] = max(iou_sore,[],2);
        unique_pick_score_id = unique(pick_score_id);
        iou = 0;
        axis_gt = axis_gt(1:size(GT_seg,1)-1,:);
        axis_gt = [0 0 0 0 0 1 0 0 0;axis_gt];
        pred_axis = zeros(size(GT_seg,1),8);
        
        for j = 1:size(unique_pick_score_id,1)
            temp_pick_id = find(pick_score_id == unique_pick_score_id(j));
            [temp_pick_max_score,tmp_p_axis_idx] = max(pick_max_score(temp_pick_id));
            iou = iou + temp_pick_max_score;
            pred_axis(unique_pick_score_id(j),:) = choose_axis(temp_pick_id(tmp_p_axis_idx),:);
        end
        iou = iou/size(GT_seg,1);
        all_class_iou = all_class_iou + iou;
        
        uvw_err = 0;
        theta_err = 0;
        phi_err = 0;
        xyz_err = 0;
        type_acc = 0;
        statistic_axis_count = 0;
        all_zeros = all(pred_axis==0,2);
        for j = 1:size(axis_gt,1)
            if all_zeros(j) == 1
                continue;
            end
            statistic_axis_count = statistic_axis_count + 1;
            tmp_axis_gt = axis_gt(j,:);
            
            tmp_axis_pred = pred_axis(j,:);
            theta_gt = tmp_axis_gt(8)*10;
            phi_gt = tmp_axis_gt(9)*10;
            type_gt = tmp_axis_gt(7);
            if type_gt == 1
                phi_gt = 0;
            elseif type_gt ==2
                theta_gt = 0;
            end
            theta_pred = tmp_axis_pred(7);
            theta_err = theta_err + abs(theta_gt-theta_pred);
            phi_pred = tmp_axis_pred(8);
            phi_err = phi_err + abs(phi_gt-phi_pred);
            uvw_gt = tmp_axis_gt(4:6);
            uvw_pred = tmp_axis_pred(4:6);
            cosine = 1-(uvw_gt * uvw_pred') / (norm(uvw_gt)*norm(uvw_pred)+0.0001);
            uvw_err = uvw_err + cosine;
            dof_gt = tmp_axis_gt(1:6);
            xyz_pred = tmp_axis_pred(:,1:3);
            
            if (type_gt~=2)
                start_p_dis = compute_dis(xyz_pred,dof_gt);
                xyz_err = xyz_err + start_p_dis;
            end
            
            if(type_gt ==0)
                if(theta_pred<10&&phi_pred<0.05)
                    type_acc = type_acc + 1;
                end
            elseif(type_gt ==1)
                if(theta_pred>10&&phi_pred<0.05)
                    type_acc = type_acc + 1;
                end
            elseif(type_gt ==2)
                if(theta_pred<10&&phi_pred>0.05)
                    type_acc = type_acc + 1;
                end
            elseif(type_gt ==3)
                if(theta_pred>10&&phi_pred>0.05)
                    type_acc = type_acc + 1;
                end
            end
            
        end
        
        pred_data.input = input_gt;
        pred_data.GT_label = label;
        pred_data.GT_axis = axis_gt;
        pred_data.pred_label = choose_idx;
        pred_data.pred_axis = pred_axis;
        pred_data.iou = iou;
        mkdir('paint')
        mkdir('paint/', temp_class)
        save(['paint/' temp_class '/' temp_class '_' num2str(i)],'pred_data');
        
        uvw_err = uvw_err/statistic_axis_count;
        theta_err = theta_err/statistic_axis_count;
        phi_err = phi_err/statistic_axis_count;
        xyz_err = xyz_err/statistic_axis_count;
        type_acc = type_acc/statistic_axis_count;
        
        all_class_uvw_err = all_class_uvw_err + uvw_err;
        all_class_theta_err = all_class_theta_err+theta_err;
        all_class_phi_err = all_class_phi_err+phi_err;
        all_class_xyz_err = all_class_xyz_err+xyz_err;
        all_class_type_acc = all_class_type_acc+type_acc;
        
        rebuild_err = [];
        unique_choose_idx = unique(choose_idx);
        for j = 1:size(unique_choose_idx,1)
            temp_first_frame = first_frame(choose_idx==unique_choose_idx(j),:);
            temp_last_frame = last_frame(choose_idx==unique_choose_idx(j),:);
            temp_rebuild_axis = choose_axis(unique_choose_idx(j),:);
            temp_rebuild_matrix = get_m(temp_rebuild_axis);
            temp_first_frame = [temp_first_frame ones(size(temp_first_frame,1),1)];
            temp_first_frame = temp_first_frame';
            temp_rebuild_frame = temp_rebuild_matrix * temp_first_frame;
            temp_rebuild_frame = temp_rebuild_frame';
            temp_rebuild_frame = temp_rebuild_frame(:,1:3);
            temp_rebuild_err = (temp_rebuild_frame - temp_last_frame) .* (temp_rebuild_frame - temp_last_frame);
            temp_rebuild_err = sum(temp_rebuild_err,2);
            temp_rebuild_err = sqrt(temp_rebuild_err+0.00000001);
            rebuild_err = [rebuild_err;temp_rebuild_err];
        end
        all_class_rebuild_error = all_class_rebuild_error + mean(rebuild_err);
    end
    all_class_iou = all_class_iou/size(filename,1);
    all_class_uvw_err = all_class_uvw_err/size(filename,1);
    all_class_theta_err = all_class_theta_err/size(filename,1);
    all_class_phi_err = all_class_phi_err/size(filename,1);
    all_class_xyz_err = all_class_xyz_err/size(filename,1);
    all_class_type_acc = all_class_type_acc/size(filename,1);
    all_class_rebuild_error = all_class_rebuild_error/size(filename,1);
    class_number = size(filename,1);
    mkdir('statistic')
    save(['statistic/' temp_class],'class_number','all_class_iou','all_class_uvw_err','all_class_theta_err','all_class_phi_err','all_class_xyz_err','all_class_type_acc','all_class_rebuild_error');
end
end

function dis = compute_dis(p,dof)
P = p;
Q1 = dof(1:3);
Q2 = dof(1:3)+dof(4:6);
dis = norm(cross(Q2-Q1,P-Q1))./norm(Q2-Q1);
end

function GT_seg = generate_seg(label)
unique_label = unique(label);
GT_seg = zeros(size(unique_label,1),size(label,1));
for i = 1:size(unique_label,1)
    GT_seg(i,label==unique_label(i)) = 1;
end
end

function iou_score = compute_score(GT_proposal,s_mat)
GT_proposal = repmat(GT_proposal,size(s_mat,1),1);
union = logical(GT_proposal+s_mat);
inter = GT_proposal.*s_mat;
iou_score = sum(inter,2)./sum(union,2);
end

function rebuild_distance = compute_rebuild_distance(matrix_sequence_pred,input_trajecotires,idx,frame_num)
n_point = size(input_trajecotires,1);
idx = mat2cell(idx,ones(n_point,1));
rebuild_distance = cellfun(@(x) compute_rebuild_distance_step(matrix_sequence_pred,input_trajecotires,x, frame_num),idx,'Unif',0);
rebuild_distance = cell2mat(rebuild_distance);
rebuild_distance = rebuild_distance';
end

function rebuild_distance = compute_rebuild_distance_step(matrix_sequence_pred,input_trajecotires,idx, frame_num)
n_frame = size(input_trajecotires,2);
rebuild_distance = 0;
for i = 1:size(idx,2)
    input = reshape(input_trajecotires(idx(i),:,1:3),n_frame,3);
    temp_rebuild_distance = [];
    temp_consine_distance = [];
    for j = 2:frame_num
        matrix = matrix_sequence_pred{j-1};
        tmp_dis = compute_rebuild_distance_one_point(matrix,input(1,:),input(j,:));
        tmp_cosine = compute_consine_distance(matrix,input(j-1,:),input(j,:),input,frame_num);
        temp_rebuild_distance = [temp_rebuild_distance tmp_dis];
        temp_consine_distance = [temp_consine_distance tmp_cosine];
    end
    xxx = mean(temp_rebuild_distance)+0.2*mean(temp_consine_distance);
    rebuild_distance = rebuild_distance + xxx;
end
rebuild_distance = rebuild_distance/size(idx,2);
end

function cosine_distance = compute_consine_distance(matrix,input1,input2,input_all,frame_num)
first_frame = input_all(1,:);
flag = input_all - repmat(first_frame,frame_num,1);
flag = sqrt(sum(flag.*flag,2)+0.00000001);
flag = mean(flag);
if (flag<0.03)
    cosine_distance = 0;
else
    rebuild = matrix * [input1 1]';
    rebuild = rebuild(1:3);
    dir1 = input_all(frame_num,:)-input_all(1,:);
    dir2 = rebuild'-input1;
    cosine_distance = 1-(dir1*dir2')/(norm(dir1)*norm(dir2)+0.0001);
end
end

function rebuild_distance = compute_rebuild_distance_one_point(matrix,input1,input2)
rebuild = matrix * [input1 1]';
rebuild = rebuild(1:3);
rebuild_distance = sqrt(sum((input2'-rebuild).*(input2'-rebuild))+0.00000001);
end

function matrix_sequence = compute_matrix_sequence(axis, frame_num)
axis_rep = repmat(axis,(frame_num - 1),1);
for i = 1:(frame_num - 1)
    axis_rep(i,7:8) = axis_rep(i,7:8)/(frame_num - 1)*i;
end
axis_rep = mat2cell(axis_rep,ones((frame_num - 1),1));
matrix_sequence = cellfun(@(x) get_m(x),axis_rep,'Unif',0);
matrix_sequence = matrix_sequence';
end

function dis = compute_trajectories_dis(input_gt,x)
temp_x = repmat(x,size(input_gt,1),1);
error = input_gt - temp_x;
error = error.*error;
error = sum(error,3);
dis = mean(error,2);
dis = dis';
end

function out = get_m (axis)
x = axis(1);
y = axis(2);
z = axis(3);
u = axis(4);
v = axis(5);
w = axis(6);
rs = axis(7);
ts = axis(8);
rs = rs * 3.1415629535897932384626 / 180;
norm = max(sqrt(u*u + v*v + w*w),1e-6);
u = u/norm;
v = v/norm;
w = w/norm;
cosaa = cos(rs);
sinaa = sin(rs);
tx = u * ts;
ty = v * ts;
tz = w * ts;
out = zeros(4,4);
out(1,1) = u*u+(v*v+w*w)*cosaa;
out(1,2) = u*v*(1-cosaa)-w*sinaa;
out(1,3) = u*w*(1-cosaa)+v*sinaa;
out(1,4) = (x*(v*v+w*w)-u*(y*v+z*w))*(1-cosaa)+(y*w-z*v)*sinaa+tx;
out(2,1) = u*v*(1-cosaa)+w*sinaa;
out(2,2) = v*v+(u*u+w*w)*cosaa;
out(2,3) = v*w*(1-cosaa)-u*sinaa;
out(2,4) = (y*(u*u+w*w)-v*(x*u+z*w))*(1-cosaa)+(z*u-x*w)*sinaa+ty;
out(3,1) = u*w*(1-cosaa)-v*sinaa;
out(3,2) = v*w*(1-cosaa)+u*sinaa;
out(3,3) = w*w+(u*u+v*v)*cosaa;
out(3,4) = (z*(u*u+v*v)-w*(x*u+y*v))*(1-cosaa)+(x*v-y*u)*sinaa+tz;
out(4,1) = 0;
out(4,2) = 0;
out(4,3) = 0;
out(4,4) = 1;

end

function enhance_axis = enhance_axis_by_theta_phi(axis)
axis_cell = mat2cell(axis,ones(size(axis,1),1));
enhance_axis_cell = cellfun(@(x) enhance_axis_by_theta_phi_step(x),axis_cell,'Unif',0);
enhance_axis = cell2mat(enhance_axis_cell);
end

function enhance_axis = enhance_axis_by_theta_phi_step(axis)
enhance_axis = axis;
if axis(7) > 0
    temp_axis= axis;
    temp_axis(7) = temp_axis(7) + 5;
    enhance_axis = [enhance_axis;temp_axis];
    temp_axis= axis;
    temp_axis(7) = temp_axis(7) - 5;
    enhance_axis = [enhance_axis;temp_axis];
end
if axis(8) > 0
    temp_axis= axis;
    temp_axis(8) = temp_axis(8) + 0.05;
    enhance_axis = [enhance_axis;temp_axis];
    temp_axis= axis;
    temp_axis(8) = temp_axis(8) - 0.05;
    enhance_axis = [enhance_axis;temp_axis];
    
        temp_axis= axis;
    temp_axis(8) = temp_axis(8) + 0.03;
    enhance_axis = [enhance_axis;temp_axis];
    temp_axis= axis;
    temp_axis(8) = temp_axis(8) - 0.03;
    enhance_axis = [enhance_axis;temp_axis];
    
        temp_axis= axis;
    temp_axis(8) = temp_axis(8) + 0.10;
    enhance_axis = [enhance_axis;temp_axis];
    temp_axis= axis;
    temp_axis(8) = temp_axis(8) - 0.10;
    enhance_axis = [enhance_axis;temp_axis];
    
        temp_axis= axis;
    temp_axis(8) = temp_axis(8) + 0.15;
    enhance_axis = [enhance_axis;temp_axis];
    temp_axis= axis;
    temp_axis(8) = temp_axis(8) - 0.15;
    enhance_axis = [enhance_axis;temp_axis];
    
end
end
