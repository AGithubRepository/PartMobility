function generate_rotate_face
path = 'test_data'
for i = 1:2000
    type = unidrnd(3);
    if type==1
        A = rand()-0.5;
        B = rand()-0.5;
        C = rand()-0.5;
        D = rand()-0.5;
        x = rand(1024,1)-0.5;
        y = rand(1024,1)-0.5;
        while(abs(C)<1e-6)
            C = rand()-0.5;
        end
        z = (D-A*x-B*y)/C;
        z = z-mean(z);
        max_z = max(z);
        min_z = min(z);
        z = z / (max_z-min_z);
        point = [x y z];
    elseif type==2
        A = rand()-0.5;
        B = rand()-0.5;
        C = rand()-0.5;
        D = rand()-0.5;
        x = rand(1024,1)-0.5;
        z = rand(1024,1)-0.5;
        while(abs(B)<1e-6)
            B = rand()-0.5;
        end
        y = (D-A*x-C*z)/B;
        y = y-mean(y);
        max_y = max(y);
        min_y = min(y);
        y = y / (max_y-min_y);
        point = [x y z];
    elseif type==3
        A = rand()-0.5;
        B = rand()-0.5;
        C = rand()-0.5;
        D = rand()-0.5;
        y = rand(1024,1)-0.5;
        z = rand(1024,1)-0.5;
        while(abs(A)<1e-6)
            A = rand()-0.5;
        end
        x = (D-B*y-C*z)/A;
        x = x-mean(x);
        max_x = max(x);
        min_x = min(x);
        x = x / (max_x-min_x);
        point = [x y z];
    end
    point = unique(point,'rows');
    if size(point,1)<1024
        continue;
    end
    idx = randperm(size(point,1));
    idx = idx(1:1024);
    point=point(idx,:);
    axis_x = rand()-0.5;
    axis_y = rand()-0.5;
    axis_z = rand()-0.5;
    axis_u = rand()-0.5;
    axis_v = rand()-0.5;
    axis_w = rand()-0.5;
    motion_type_percent = unidrnd(10);
    
    if motion_type_percent==1
        motion_type=1;
    elseif motion_type_percent>=2&&motion_type_percent<=3
        motion_type=3;
    elseif motion_type_percent>=4&&motion_type_percent<=8
        motion_type=2;
    elseif motion_type_percent>=9&&motion_type_percent<=10
        motion_type=4;
    end
    
    
    if motion_type == 1
        rs = 0;
        ts = 0;
    elseif motion_type == 2
        rs = rand()*5 + 5;
        ts = 0;
    elseif motion_type == 3
        rs = 0;
        ts = rand()*0.02 + 0.02;
    elseif motion_type == 4
        rs = rand()*5 + 5;
        ts = rand()*0.02 + 0.02;
    end
    temp_m = eye(4);
    m = get_m(axis_x,axis_y,axis_z,axis_u,axis_v,axis_w,rs,ts);
    input = zeros(1024,11,6);
    for j = 1:11
        input(:,j,1:3) = point;
        temp_point = point;
        temp_point = [temp_point ones(1024,1)];
        temp_point = temp_point';
        temp_point = m*temp_point;
        temp_point = temp_point';
        temp_point = temp_point(:,1:3);
        point = temp_point;
    end
    center_frame = input(:,(11+1)/2,1:3);
    max_xxyyzz = max(center_frame,[],1);
    min_xxyyzz = min(center_frame,[],1);
    t_center = min_xxyyzz + (max_xxyyzz - min_xxyyzz) / 2;
    center = repmat(t_center,1024,11,1);
    axis_x = axis_x - t_center(1,1,1);
    axis_y = axis_y - t_center(1,1,2);
    axis_z = axis_z - t_center(1,1,3);
    
    input(:,:,1:3) = input(:,:,1:3) - center + rand(1024,11,3)*0.01;
    for j = 1:10
        input(:,j,4:6) = input(:,j+1,1:3) - input(:,j,1:3);
    end
    all_m = m*m*m*m*m*m*m*m*m*m
    xxx = reshape(input(:,1,1:3),1024,3);
    yyy = reshape(input(:,11,1:3),1024,3);
    [abspose_m,t] = abspose(xxx',yyy');
    abspose_m = [abspose_m t];
    abspose_m = [abspose_m;[0 0 0 1]]
    
    %     figure(1)
    %     plot3(input(:,1,1),input(:,1,2),input(:,1,3),'*')
    %     hold on;
    %     for j = 1:1024
    %     plot3(input(j,:,1),input(j,:,2),input(j,:,3));
    %     hold on;
    %     end
    %     axis equal;
    %     close(figure(1));
    
    data.input = input;
    data.T = 11;
    data.m = all_m;
    data.axis = [axis_x,axis_y,axis_z,axis_u,axis_v,axis_w,motion_type,rs*10,ts*10];
    
    save([path '/data_' num2str(i,'%04d')],'data');
end




end

function out = get_m (x,y,z,u,v,w,rs,ts)
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

function pc = getpc(num)

x = rand(num,1);
y = rand(num,1);
z = zeros(num,1);
pc = [x y z];


end