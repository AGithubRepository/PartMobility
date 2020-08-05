function cal_pred_acc_s_data_2
path = 'pred';
classfile = dir(fullfile(path,'*'));
classname = {classfile.name}';
classname = classname(3:end);
all_rebuild_loss = 0;
all_uvw_loss = 0;
all_theta_loss = 0;
all_phi_loss = 0;
all_xyz_loss = 0;
all_type_loss = 0;

all_rebuild_count = 0;
all_uvw_count = 0;
all_theta_count = 0;
all_phi_count = 0;
all_xyz_count = 0;
all_type_count = 0;

pred_axis_num = 64;
p_number = 2048;
input_frame = 2;
choose_rate = [];
for c_k = 1:size(classname,1)
    temp_class = classname{c_k};
    classpath = [path '/' temp_class];
    file = dir(fullfile(classpath,'*.mat'));
    filename = {file.name}';
    
    all_class_iou = 0;
    
    for i = 1:size(filename,1)
        load([classpath '/' filename{i}])
        % input
        label = reshape(label,p_number,1);
        axis_uvw = reshape(axis_uvw,pred_axis_num,3);
        axis_xyz = reshape(axis_xyz,pred_axis_num,3);
        theta = reshape(theta,pred_axis_num,1)/3.1415629535897932384626*180;
        theta(theta<10) = 0;
        phi = reshape(phi,pred_axis_num,1);
        phi(phi<0.05) = 0;
        input_gt = reshape(input_gt,p_number,input_frame,6);
        
        input_gt = input_gt(:,:,1:3);
        input_gt_cell = mat2cell(input_gt,ones(p_number,1));
        trajectories_dis = cellfun(@(x) compute_trajectories_dis(input_gt,x),input_gt_cell,'Unif',0);
        trajectories_dis = cell2mat(trajectories_dis);
        [~,sort_b] = sort(trajectories_dis,2);
        sort_b = sort_b(:,1:3);
        m_gt = reshape(m_gt,80,4);
        abspose_axis = reshape(abspose_axis,pred_axis_num,3);
        abspose_angle = reshape(abspose_angle,pred_axis_num,1)/3.1415926535897*180;
        abspose_trans = reshape(abspose_trans,pred_axis_num,1);
        abspose_m = reshape(abspose_m,pred_axis_num,4,4);
        axis_gt = reshape(axis_gt,10,9);
        axis_pred_all = [axis_xyz axis_uvw theta phi];
        no_movement_idx = find(theta<10&phi<0.05);
        axis_pred_all(no_movement_idx,:) = [];
        axis_pred_all = [axis_pred_all;0 0 0 0 0 1 0 0];
        
        axis_pred_all_cell = mat2cell(axis_pred_all,ones(size(axis_pred_all,1),1));
        matrix_pred = cellfun(@(x) get_m(x),axis_pred_all_cell,'Unif',0);
        rebuild_distance = cellfun(@(x) compute_rebuild_distance(x,input_gt,sort_b),axis_pred_all_cell,'Unif',0);
        rebuild_distance = cell2mat(rebuild_distance);
        
        [val,idx] = min(rebuild_distance);
        statistic = tabulate(idx);
        axis_idx = statistic(:,1);
        count = statistic(:,2);
        first_frame = reshape(input_gt(:,1,1:3),p_number,3);
        %     figure(1)
        %     for j = 1:size(count,1)
        %         temp_first_frame = first_frame(idx==axis_idx(j),:);
        %         plot3(temp_first_frame(:,1),temp_first_frame(:,2),temp_first_frame(:,3),'o')
        %         hold on;
        %         axis equal;
        %     end
        %     close(figure(1));
        loop_axis_idx = axis_idx;
        loop_count = count;
        choose_axis = [];
        choose_axis_idx = [];
        while(true)
            if (~isempty(choose_axis))
                [~,max_idx] = max(loop_count);
                if(loop_count(max_idx)<10)
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
                if(xxx(7)>10)
                    dis_thres = 0.03;
                else
                    dis_thres = 0.01;
                end
                ratio = sum(distance_err<dis_thres)/size(belonging_idx,2)
                
                if(ratio>0.5)
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
        
        
        choose_axis_cell = mat2cell(choose_axis,ones(size(choose_axis,1),1));
        
        choose_rebuild_distance = cellfun(@(x) compute_rebuild_distance(x,input_gt,sort_b),choose_axis_cell,'Unif',0);
        choose_rebuild_distance = cell2mat(choose_rebuild_distance);
        
        [choose_val,choose_idx] = min(choose_rebuild_distance);
        choose_statistic = tabulate(choose_idx);
        pred_label = choose_statistic(:,1);
        pred_count = choose_statistic(:,2);
        first_frame = reshape(input_gt(:,1,1:3),p_number,3);
        
        all_pred_label = choose_idx(sort_b);
        choose_idx = mode(all_pred_label,2);
        
        
        
%         figure(1)
%         for j = 1:size(pred_label,1)
%             temp_first_frame = first_frame(choose_idx==pred_label(j),:);
%             plot3(temp_first_frame(:,1),temp_first_frame(:,2),temp_first_frame(:,3),'o')
%             hold on;
%             axis equal;
%         end
%         close(figure(1))
        
        GT_seg = generate_seg(label);
        pred_seg = generate_seg(choose_idx);
        GT_seg_cell = mat2cell(GT_seg,ones(size(GT_seg,1),1));
        
        iou_sore = cellfun(@(x) compute_score(x,pred_seg),GT_seg_cell,'Unif',0);
        iou_sore = iou_sore';
        iou_sore = cell2mat(iou_sore);
        [pick_max_score,pick_score_id] = max(iou_sore,[],2);
        unique_pick_score_id = unique(pick_score_id);
        iou = 0;
        axis_gt = axis_gt(size(GT_seg,1)-1,:);
        axis_gt = [0 0 0 0 1 0 0 0 0;axis_gt];
        pred_axis = zeros(size(GT_seg,1),8);
        
        for j = 1:size(unique_pick_score_id,1)
            temp_pick_id = find(pick_score_id == unique_pick_score_id(j));
            [temp_pick_max_score,tmp_p_axis_idx] = max(pick_max_score(temp_pick_id));
            iou = iou + temp_pick_max_score;
            pred_axis(temp_pick_id,:) = choose_axis(temp_pick_id(tmp_p_axis_idx),:);
        end
        iou = iou/size(GT_seg,1);
        all_class_iou = all_class_iou + iou;
        
        uvw_cosine = 0;
        uvw_count = 0;
        theta_err = 0;
        theta_count = 0;
        phi_err = 0;
        phi_count = 0;
        xyz_err = 0;
        xyz_count = 0;
        type_acc = 0;
        type_count = 0;
        all_zeros = all(pred_axis==0,2)
        for j = 1:size(axis_gt,1)
            if all_zeros(j) == 1
                continue;
            end
            tmp_axis_gt = axis_gt(j,:);
            tmp_axis_pred = pred_axis(j,:);
            theta_gt = tmp_axis_gt(8);
            theta_pred = tmp_axis_pred(7);
            theta_err = theta_err + abs(theta_gt-theta_pred);
            theta_count = theta_count + 1;

            phi_gt = tmp_axis_gt(9);
            phi_pred = tmp_axis_pred(8);
            phi_err = phi_err + abs(phi_gt-phi_pred);
            phi_count = phi_count + 1;
            
            uvw_gt = tmp_axis_gt(4:6);
            uvw_pred = tmp_axis_pred(4:6);
            cosine = (uvw_gt * uvw_pred') / (norm(uvw_gt)*norm(uvw_pred));
    if (phi_gt > 0.05 || theta_gt > 10)
        all_uvw_loss = all_uvw_loss + cosine;
        all_uvw_count = all_uvw_count + 1;
    end


    

    
    
    xyz_pred = axis_pred(:,1:3);
    dof_gt = axis_gt(1:6);
    if (phi_gt > 0.05 || theta_gt > 10)
        start_p_dis = compute_dis(xyz_pred,dof_gt);
        all_xyz_loss = all_xyz_loss + start_p_dis;
        all_xyz_count = all_xyz_count + 1;
    end
    
    type_gt = axis_gt(7);
    
    if(type_gt ==1)
        if(theta_pred<10&&phi_pred<0.05)
            all_type_loss = all_type_loss + 1;
        end
    elseif(type_gt ==2)
        if(theta_pred>10&&phi_pred<0.05)
            all_type_loss = all_type_loss + 1;
        end
    elseif(type_gt ==3)
        if(theta_pred<10&&phi_pred>0.05)
            all_type_loss = all_type_loss + 1;
        end
    elseif(type_gt ==4)
        if(theta_pred>10&&phi_pred>0.05)
            all_type_loss = all_type_loss + 1;
        end
    end

        end
    end
    all_class_iou = all_class_iou/size(filename,1);
end
rebuild_err = all_rebuild_loss/all_rebuild_count;
uvw_cosine = all_uvw_loss/all_uvw_count;
theta_err = all_theta_loss/all_theta_count;
phi_err = all_phi_loss/all_phi_count;
xyz_err = all_xyz_loss/all_xyz_count;
type_acc = all_type_loss/all_type_count;

fprintf('rebuild_err is %f\n',all_rebuild_count);
fprintf('uvw_cosine is %f\n',all_uvw_count);
fprintf('theta_err is %f\n',all_theta_count);
fprintf('phi_err is %f\n',all_phi_count);
fprintf('xyz_err is %f\n',all_xyz_count);
fprintf('type_acc is %f\n',all_type_count);

fprintf('rebuild_err is %f\n',rebuild_err);
fprintf('uvw_cosine is %f\n',uvw_cosine);
fprintf('theta_err is %f\n',theta_err);
fprintf('phi_err is %f\n',phi_err);
fprintf('xyz_err is %f\n',xyz_err);
fprintf('type_acc is %f\n',type_acc);

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

function dis = compute_trajectories_dis(input_gt,x)
temp_x = repmat(x,size(input_gt,1),1);
error = input_gt - temp_x;
error = error.*error;
error = sum(error,3);
dis = mean(error,2);
dis = dis';
end

function rebuild_distance = compute_rebuild_distance(axis_pred_all_cell,input,idx)
n_point = size(input,1);
idx = mat2cell(idx,ones(n_point,1));
rebuild_distance = cellfun(@(x) compute_rebuild_distance_step(axis_pred_all_cell,input,x),idx,'Unif',0);
rebuild_distance = cell2mat(rebuild_distance);
rebuild_distance = rebuild_distance';
end

function rebuild_distance = compute_rebuild_distance_step(axis_pred_all_cell,input_all,idx)
n_frame = size(input_all,2);
rebuild_distance = 0;
temp_axis_pred_all_cell=axis_pred_all_cell;
for i = 1:size(idx,2)
    input = reshape(input_all(idx(i),:,1:3),n_frame,3);
    input_cell = mat2cell(input,ones(n_frame,1));
    temp_rebuild_distance = [];
    for j = 2:2
        temp_axis_pred_all_cell(7:8) = axis_pred_all_cell(7:8)/10*(j-1);
        matrix = get_m(temp_axis_pred_all_cell);
        temp_rebuild_distance = [temp_rebuild_distance compute_rebuild_distance_one_point(matrix,input(1,:),input(j,:))];
    end
    temp_rebuild_distance = max(temp_rebuild_distance);
    rebuild_distance = rebuild_distance + temp_rebuild_distance;
end
rebuild_distance = rebuild_distance/size(idx,2);
end

function rebuild_distance = compute_rebuild_distance_one_point(matrix,input1,input2)
rebuild = matrix * [input1 1]';
rebuild = rebuild(1:3);
rebuild_distance = sqrt(sum((input2'-rebuild).*(input2'-rebuild))+0.00000001);
end

function dis = compute_dis(p,dof)
P = p;
Q1 = dof(1:3);
Q2 = dof(1:3)+dof(4:6);
dis = norm(cross(Q2-Q1,P-Q1))./norm(Q2-Q1);
end

function dis = compute_start(matrix,start_p)
matrix = reshape(matrix,4,4);
R = matrix(1:3,1:3);
b = matrix(1:3,4:4);
dis = (R-eye(3)) * start_p' + b;
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