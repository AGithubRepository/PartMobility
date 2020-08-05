function cal_pred_acc_F_data
path = 'pred';
file = dir(fullfile(path,'*.mat'));
filename = {file.name}';

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
choose_rate = [];
for i = 22:size(filename,1)
    load([path '/' filename{i}])
    % input
    axis_uvw = reshape(axis_uvw,pred_axis_num,3);
    axis_xyz = reshape(axis_xyz,pred_axis_num,3);
    theta = reshape(theta,pred_axis_num,1)/3.1415629535897932384626*180;
    theta(theta<10) = 0;
    phi = reshape(phi,pred_axis_num,1);
    phi(phi<0.05) = 0;
    input_gt = reshape(input_gt,p_number,11,6);
    m_gt = reshape(m_gt,80,4);
    abspose_axis = reshape(abspose_axis,pred_axis_num,3);
    abspose_angle = reshape(abspose_angle,pred_axis_num,1)/3.1415926535897*180;
    abspose_trans = reshape(abspose_trans,pred_axis_num,1);
    abspose_m = reshape(abspose_m,pred_axis_num,4,4);
    axis_gt = reshape(axis_gt,10,9);
    %     norm = max(sqrt(axis_uvw(:,1).*axis_uvw(:,1) + axis_uvw(:,2).*axis_uvw(:,2) + axis_uvw(:,3).*axis_uvw(:,3)),1e-6);
    %     axis_uvw(:,1) = axis_uvw(:,1)./norm;
    %     axis_uvw(:,2) = axis_uvw(:,2)./norm;
    %     axis_uvw(:,3) = axis_uvw(:,3)./norm;
    %
    %     norm2 = max(sqrt(axis_gt(:,4).*axis_gt(:,4) + axis_gt(:,5).*axis_gt(:,5) + axis_gt(:,6).*axis_gt(:,6)),1e-6);
    %     axis_gt(:,4) = axis_gt(:,4)./norm2;
    %     axis_gt(:,5) = axis_gt(:,5)./norm2;
    %     axis_gt(:,6) = axis_gt(:,6)./norm2;
    
    % choose
    axis_pred_all = [axis_xyz axis_uvw theta phi];
%     axis_pred_all = [axis_xyz axis_uvw abspose_angle abspose_trans];
%     axis_pred_all = [repmat(axis_gt(1:3),32,1) abspose_axis abspose_angle abspose_trans];
%     axis_pred_all = [repmat(axis_gt(1:6),32,1) repmat(axis_gt(8:9),32,1)];
%     axis_pred_all = [repmat(axis_gt(1:3),32,1) axis_uvw theta phi];
    axis_pred_all_cell = mat2cell(axis_pred_all,ones(pred_axis_num,1));
    matrix_pred = cellfun(@(x) get_m(x),axis_pred_all_cell,'Unif',0);
    input_frame_s = reshape(input_gt(:,1,1:3),p_number,3);
    input_frame_e = reshape(input_gt(:,11,1:3),p_number,3);
    input_frame_s = [input_frame_s ones(p_number,1)];
    input_frame_e_pred = cellfun(@(x) x*input_frame_s',matrix_pred,'Unif',0);
    input_frame_e_pred = cellfun(@(x) x',input_frame_e_pred,'Unif',0);
    input_frame_e_pred = cellfun(@(x) x(:,1:3),input_frame_e_pred,'Unif',0);
    input_frame_s = input_frame_s(:,1:3);
    distance = cellfun(@(x) sqrt(sum((x-input_frame_e).*(x-input_frame_e),2)),input_frame_e_pred,'Unif',0);
    distance = cellfun(@(x) x',distance,'Unif',0);
    distance = cell2mat(distance);
    [val,idx] = min(distance);
    statistic = tabulate(idx);
    count = statistic(:,2);

    [val_count,idx_count] = max(count);
    choose_rate = [choose_rate;statistic(idx_count,3)];
    % loss
    rebuild_loss = mean(distance(idx_count,:));
    all_rebuild_loss = all_rebuild_loss + rebuild_loss;
    all_rebuild_count = all_rebuild_count + 1;
    
    axis_pred = axis_pred_all(idx_count,:);
    
    theta_gt = axis_gt(8);
    theta_pred = axis_pred(7);
    if (theta_gt > 10)
        all_theta_loss = all_theta_loss + abs(theta_gt-theta_pred);
        all_theta_count = all_theta_count + 1;
    end
    
    phi_gt = axis_gt(9);
    phi_pred = axis_pred(8);
    if (phi_gt > 0.05)
        all_phi_loss = all_phi_loss + abs(phi_gt-phi_pred);
        all_phi_count = all_phi_count + 1;
    end
    
    uvw_gt = axis_gt(4:6);
    uvw_pred = axis_pred(4:6);
    cosine = (uvw_gt * uvw_pred') / norm(uvw_gt)/norm(uvw_pred);
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
    all_type_count = all_type_count + 1;
    
        figure(1)
        for j = 1:1024
            plot3(input_gt(j,:,1),input_gt(j,:,2),input_gt(j,:,3));
            hold on;
        end
        axis equal;
        quiver3(axis_gt(:,1),axis_gt(:,2),axis_gt(:,3),axis_gt(:,4),axis_gt(:,5),axis_gt(:,6))
        hold on;
        quiver3(axis_xyz(:,1),axis_xyz(:,2),axis_xyz(:,3),axis_uvw(:,1),axis_uvw(:,2),axis_uvw(:,3))
    
        figure(2)
        quiver3(axis_gt(:,1),axis_gt(:,2),axis_gt(:,3),axis_gt(:,4),axis_gt(:,5),axis_gt(:,6))
        hold on;
        quiver3(axis_xyz(:,1),axis_xyz(:,2),axis_xyz(:,3),axis_uvw(:,1),axis_uvw(:,2),axis_uvw(:,3))
        %quiver3(repmat(axis_gt(:,1),32,1),repmat(axis_gt(:,2),32,1),repmat(axis_gt(:,3),32,1),abspose_axis(:,1),abspose_axis(:,2),abspose_axis(:,3))
        axis equal
        close(figure(1));
        close(figure(2));
end

rebuild_err = all_rebuild_loss/all_rebuild_count;
uvw_cosine = all_uvw_loss/all_uvw_count;
theta_err = all_theta_loss/all_theta_count;
phi_err = all_phi_loss/all_phi_count;
xyz_err = all_xyz_loss/all_xyz_count;
type_acc = all_type_loss/all_type_count;

fprintf('rebuild_err is %f\n',rebuild_err);
fprintf('uvw_cosine is %f\n',uvw_cosine);
fprintf('theta_err is %f\n',theta_err);
fprintf('phi_err is %f\n',phi_err);
fprintf('xyz_err is %f\n',xyz_err);
fprintf('type_acc is %f\n',type_acc);

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