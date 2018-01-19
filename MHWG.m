%Input MH WG steps here
%Define proposals: (mu,sigma)
% qsig=@(nu,x) (((2)^(-nu/2)/gamma(nu/2))*x^(-nu/2-1)*exp(-1/(2*x)));
qsig=@ (nu,x) chi2rnd(x);
qmu=@(mu,sigma) normrnd(mu,sigma);

%Define proposals: ETA={c, g_r, s, theta, T} EXCLUDING T
        
%Updating scaling paramter s: q=N(sj_prev, sigma_sj)
qs=@(sj_n_1,sigma_sj) normrnd(sj_n_1,sigma_sj);
qs_X = @(X,mu,sigma) normpdf(X,mu,sigma);

%Updating location paramter c: q=N(cj_prev, sigma_cjI)
qc=@(cj,sigma_sj) mvnrnd(cj,sigma_sj);  
qc_X = @(X,mu,sigma) mvnpdf(X,mu,sigma);

%Updating rotation paramter theta: q=UNI(0,PI)
%qtheta=unifrnd(0,pi);
qtheta_X=@(X) unifpdf(X,0,pi);

%Updating random pure paramter g_r: If T=circle(2): No action required
%                                   If T=elipse(1): q=E1=UNI(1.005;1.8)= (a,b)
g_r_X=@(X) unifpdf(X,1.005,1.8);

%Updating template parameter T: T=DiscreteUNI(1,2)
q_T=@() unidrnd(2);
q_T_X=@(X) unidpdf(X,2);

%Updating gamma: q=LogNormal(alpha_i,delta_i);
q_gamma=@(mu,sigma) lognrnd(mu,sigma);
q_gamma_X=@(X,mu,sigma) lognpdf(X,mu,sigma);

%Initialise sampling constants

% burn=500;
%Use struct

%Initialise sampler
size1=10*size0; %From image pre-processing
fscale3=transpose(parameters(:,1)); % Tranpose to 1 by m (#objects) (row,column)
fmean3=transpose(parameters(:,6));  % Tranpose to 1 by m (#objects)
fvar3=(transpose(parameters(:,7)));  % Tranpose to 1 by m (#objects)
floc3=transpose(parameters(:,2:3)); % Tranpose to 2 by m (#objects)
ftemplate3=transpose(parameters(:,5)); % Tranpose to 1 by m (#objects) (row,column)
frotation3=transpose(parameters(:,4)); % Tranpose to 1 by m (#objects)
feccentric3=transpose(parameters(:,8)); %Transpose to 1 by m 
sigmasqr=1;

nSamples=5000;
% Initialise all sample vectors :mu, sigma, s, c, T, theta, g_r, gamma
% (alpha, delta), Nj, m
zz=size(fmean3,2);
muSample=zeros(nSamples,size1);
muSample(1,1:zz)=fmean3(1,:);
sigmaSample=zeros(nSamples,size1);
sigmaSample(1,1:zz)=fvar3(1,:);

sSample=zeros(nSamples,size1);
sSample(1,1:zz)=fscale3(1,:);
sigma_sj0=sSample(1,:);

c_xSample= zeros(nSamples,size1);
c_xSample(1,1:zz)=floc3(1,:);
c_ySample= zeros(nSamples,size1);
c_ySample(1,1:zz)=floc3(2,:);

% cSample=struct('C',floc3);
% cj_sample=zeros(2,size1);


TSample=ones(nSamples,size1);
TSample(1,1:zz)=ftemplate3(1,:);

thetaSample=ones(nSamples,size1);
thetaSample(1,1:zz)=abs((pi/180)*frotation3(1,:)); %Convert to radians

g_rSample=zeros(nSamples,size1);
for i=1:zz
    g_rSample(1,i)=1/(1-feccentric3(1,i)^2)^(1/4);
end


alpha1_sample=ones(nSamples,1);
alpha2_sample=ones(nSamples,1);
delta1_sample=ones(nSamples,1);
delta2_sample=ones(nSamples,1);
alpha1_sample(1,1)=unifrnd(3,5);
alpha2_sample(1,1)=unifrnd(3,5);
delta1_sample(1,1)=unifrnd(1,1.5);
delta2_sample(1,1)=unifrnd(1,1.5);

gamma_sample=ones(nSamples,2);
gamma_sample(1,1)=q_gamma(alpha1_sample(1,1),delta1_sample(1,1));
gamma_sample(1,2)=q_gamma(alpha2_sample(1,1),delta2_sample(1,1));

NjSample=zeros(nSamples,size1);
NjSample(1,1:zz)=fscale3(1,:);
n=sum(NjSample(1,:));
t=1;
m_sample=ones(nSamples,1);
m_sample(1,1)=size0;

m_updated_ind=0;

while t<nSamples+1;
    if m_updated_ind==1
        t=t+1;
        latest_m=m_sample(t-1,1);
    end
    if m_updated_ind==0 && t>1
        latest_m=m_sample(t-1,1);
    end
    if t==1
        t=t+1;
        latest_m=size0;
    end

    if latest_m>0
        for j=1:latest_m %For every object at time t
            [f_n,f_m]=size(f);
            sMax=f_n*f_m;
            sum_f=0;
            sum_f_star=0;
            %Gibbs steps
            %B1: Updating mu and sigma
            sigmaStar=qsig(NjSample(t-1,j)-1,sigmaSample(t-1,j)); %Draw proposal sigma from I/CHI2(Nj-1,sigma2)
            muStar=qmu(muSample(t-1,j),sigmaStar/sSample(t-1,j)); %Draw proposal mu from N(yj_bar, sigmaj/Nj)

            %B1: Updating ETA={c, g_r, s, theta, T} EXCLUDING T

            %Updating scaling paramter s: q=N(sj_prev,sigma_sj)
    %               ,sigma_sj=1/10sj_0, sj_0=estimated scale for each object    
            sStar=qs(sSample(t-1,j),sigma_sj0(1,j)/10); 

            %Updating location paramter c: q=N(cj_prev, sigma_cjI)
            x_std=std(floc3(:,1))^2;
            y_std=std(floc3(:,2))^2;
            sigma_cj0=[x_std, 0;0, y_std];
            %end1=size(cSample);
            latest_c=[c_xSample(t-1,j);c_ySample(t-1,j)];
            cStar=qc(latest_c,sigma_cj0/10); 

            %Updating rotation paramter theta: q=UNI(0,PI)
            thetaStar=unifrnd(0,pi); 

            %Updating random pure paramter g_r: If T=circle: SKIP
            %                                   If T=elipse: q=E1=UNI(1.005; 1.8)
            Tcurrent=TSample(t-1,j);
            % 1's are ellipses and 2's are circles
            if Tcurrent==1 
                g_rStar=unifrnd(1.005,1.8);
            else
                g_rStar=0;
            end   
            %Calculate accpetance probabilities 
            for f_i=1:f_n
                for f_j=1:f_m
                    item1=1/(2*sigmaSample(t-1,j));
                    item2=f(f_i,f_j);
                    item3=muSample(t-1,j);
                    sum_f=sum_f+(item1*(item2-item3));
                    item1star=1/(2*sigmaStar);
                    item2star=f(f_i,f_j);
                    item3star=muStar;
                    sum_f_star=sum_f_star+(item1star*(item2star-item3star));
                end
            end
            log_like_curr=sum_f;
            log_like_hats=sum_f_star;
    %       Steps for B1 Acceptance probability;
    %       Acceptance probabilities for MU and SIGMA
            s_j=sigmaSample(1,j);
            item1=(1/sigmaSample(t-1,j));
            item2=1/sSample(t-1,j);
            item3=-1/(2*sigmaSample(t-1,j));
            item4=(sSample(t-1,j)-1);
            item5=muSample(t-1,j)-mean(muSample(1:t-1,j));

            qcurr=item1^item2*...
                exp(item3*...
                (item4*s_j+n*(item5)^2));

            item1star=(1/sigmaStar);
            item2star=1/sStar;
            item3star=-1/(2*sigmaStar);
            item4star=(sStar-1);
            item5star=muStar-mean(muSample(1:t-1,j));

            qhats=item1star^item2star*...
                exp(item3star*...
                (item4star*s_j*(item5star)^2));
            pi1=prod(sigmaSample(t-1,1:latest_m));
            pi2=sigmaSample(t-1,j)*sigmaStar;
            pi_mu_sighats=1/(pi1/pi2);
            pi3=prod(sigmaSample(t-1,1:latest_m));
            pi_mu_sigcurr=1/(pi3);
            if qcurr==0
                qcurr=normrnd(0,0.00005);
            end
            if qhats==0
                qhats=normrnd(0,0.00005);
            end
            i1=qcurr*log_like_hats*pi_mu_sighats;
            i2=qhats*log_like_curr*pi_mu_sigcurr;
            acceptance_B1=(i1/i2); 

            %Acceptance probabilities for Eta excluding T
            qhats_s=qs_X(sStar,sStar,sigma_sj0(1,j)/10);
            holder_s=unifpdf(sSample(t-1,1:latest_m),0,sMax);
            pi_s_hats=unifpdf(sStar,0,sMax)*prod(holder_s)/holder_s(j);
            qcurr_s=qs_X(sSample(t-1,j),sSample(t-1,j),sigma_sj0(1,j)/10);
            pi_s=prod(holder_s);

            s_eta_hats=sqrt(sum(sSample(t-1,1:latest_m))-sSample(t-1,j)+sStar);
            s_eta=sum(sSample(t-1,1:latest_m));
            qhats_c=qc_X(cStar,cStar,sigma_cj0/10);

            chats1=-gamma_sample(t-1,1);
            chats2=gamma_sample(t-1,2);
            ch1=chats1*latest_m;
            ch2=chats2*s_eta_hats;
            pi_c_hats=exp(ch1-ch2);
            qcurr_c=qc_X(latest_c,latest_c,sigma_cj0/10);
            pi_c=exp(-gamma_sample(t-1,1)*latest_m-gamma_sample(t-1,2)*s_eta);

            qhats_theta=qtheta_X(thetaStar);
            holder_theta=unifpdf(thetaSample(t-1,1:latest_m),0,pi);
            pi_theta_hats=unifpdf(thetaStar,0,pi)*prod(holder_theta)/holder_theta(j);
            qcurr_theta=qtheta_X(thetaSample(t-1,j));
            pi_theta=prod(holder_theta);

            if g_rStar==0 
                qhats_gr=1;
            else
                qhats_gr=g_r_X(g_rStar);
            end
            holder_gr=ones(1,latest_m);
            for i= 1:latest_m
                if g_rSample(t-1,i)<1.005 %(e<0.15)
                    holder_gr(i)=1;
                else
                    holder_gr(i)=unifpdf(g_rSample(t-1,i),1.005,1.8);
                end
            end
            pi_gr_hats=unifpdf(g_rStar,1.005,1.8)*prod(holder_gr)/holder_gr(j);
            qcurr_gr=g_r_X(g_rSample(t-1,j));
            pi_gr=prod(holder_gr);

            acceptance_s=(qcurr_s*pi_s_hats/qhats_s*pi_s);
            acceptance_c=(qcurr_c*pi_c_hats/qhats_c*pi_c);
            acceptance_theta=(qcurr_theta*pi_theta_hats/qhats_theta*pi_theta);
            acceptance_gr=(qcurr_gr*pi_gr_hats/qhats_gr*pi_gr);
            acceptance_all=acceptance_s*acceptance_c*acceptance_theta*acceptance_gr;


            u_b1=unifrnd(0,1);
            u_b2=unifrnd(0,1);


            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %Update paramters mu, sigma, eta (excl. T)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


            if acceptance_B1 > u_b1
                muSample(t,j)=muStar;
                sigmaSample(t,j)=sigmaStar;
            else
                muSample(t,j)=muSample(t-1,j);
                sigmaSample(t,j)=sigmaSample(t-1,j);
            end
            if acceptance_all >= u_b2
                sSample(t,j)=sStar;
                NjSample(t,j)=sStar;
                c_xSample(t,j)=cStar(1);
                c_ySample(t,j)=cStar(2);
                thetaSample(t,j)=thetaStar;
                g_rSample(t,j)=g_rStar;
            else
                sSample(t,j)=sSample(t-1,j);
                NjSample(t,j)=NjSample(t-1,j);
                c_xSample(t,j)=c_xSample(t-1,j);
                c_ySample(t,j)=c_ySample(t-1,j);
                thetaSample(t,j)=thetaSample(t-1,j);
                g_rSample(t,j)=g_rSample(t-1,j);
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %B2: Updating T     
            TStar=q_T();
            if TStar == TSample(t-1,j)
                u_T=g_rSample(t,j);
                v_T=u_T;
            else
                u_T=g_rSample(t,j);
                if TStar == 1
                    v_T=unifrnd(1.005,1.8);
                else
                    v_T=0;
                end
            end
            %Calculate r=acceptance probability
    %         sum_f=0;
    %         sum_f_star=0;
    %         for f_i=1:f_n
    %             for f_j=1:f_m
    %                 sum_f=sum_f+(-1/(2*sigmaSample(t,j))*(f(f_i,f_j)-muSample(t,j)));
    %                 sum_f_star=sum_f_star+(-1/(2*sigmaStar)*(f(f_i,f_j)-muStar));
    %             end
    %         end
    %         log_like_curr=sum_f;
    %         log_like_hats=sum_f_star;
    %         pstar_That=(1/sigmaSample(t,j))^sSample(t,j)*...
    %             exp(-1/(2*sigmaSample(t,j))*...
    %             ((sSample(t,j)-1)*s_j+n*(muSample(1,j)-muSample(t,j))^2))*...
    %             qs_X(sSample(t,j),sSample(t,j),sigma_sj0(1,j)/10)*...
    %             qc_X(cSample(end1+1).C,cSample(end1+1).C,sigma_cj0/10)*...
    %             qtheta_X(thetaSample(t,j))*...
    %             g_r_X(v_T)*...
    %             log_like_curr;
    %         pstar_T=(1/sigmaSample(t,j))^sSample(t,j)*...
    %             exp(-1/(2*sigmaSample(t,j))*...
    %             ((sSample(t,j)-1)*s_j+n*(muSample(1,j)-muSample(t,j))^2))*...
    %             qs_X(sSample(t,j),sSample(t,j),sigma_sj0(1,j)/10)*...
    %             qc_X(cSample(end1+1).C,cSample(end1+1).C,sigma_cj0/10)*...
    %             qtheta_X(thetaSample(t,j))*...
    %             g_r_X(u_T)*...
    %             log_like_curr;
            if v_T<1.005
                pstar_THat=(pi_gr)/holder_gr(j)*1*pi_s*pi_c*pi_theta;
            else 
                pstar_THat=(pi_gr)/holder_gr(j)*unifpdf(v_T,1.005,1.8)*pi_s*pi_c*pi_theta;
            end;
            if u_T<1.005
                pstar_Tcurr=(pi_gr)/holder_gr(j)*1;
            else
                pstar_Tcurr=(pi_gr)/holder_gr(j)*unifpdf(u_T,1.005,1.8);
            end;
            r_A=pstar_THat*q_T_X(TSample(t-1,j))*g_r_X(u_T);
            r_B=pstar_Tcurr*q_T_X(TStar)*g_r_X(v_T);
            acceptance_T=r_A/r_B;

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %Update T
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            u_T=unifrnd(0,1);
            if acceptance_T >= u_T
                TSample(t,j)=TStar;
            else
                TSample(t,j)=TSample(t-1,j);
            end      

            %B4: Updating Gamma=LN(alpha_i,delta_i) with
            % alpha_1=alpha_2=4 and delta_i ~ UNI(1,1.5)

            alpha1=unifrnd(3,5);
            alpha2=unifrnd(3,5);
            delta1=unifrnd(1,1.5);
            delta2=unifrnd(1,1.5);

            gamma1Star=q_gamma(alpha1,delta1);
            gamma2Star=q_gamma(alpha2,delta2);

    %        Calculate acceptance probability;
            qhats_gamma1=q_gamma_X(gamma1Star,alpha1,delta1);
            qhats_gamma2=q_gamma_X(gamma2Star,alpha2,delta2);
            qcurr_gamma1=q_gamma_X(gamma_sample(t-1,1),alpha1,delta1); 
            qcurr_gamma2=q_gamma_X(gamma_sample(t-1,2),alpha2,delta2);
            pi_gamma=lognpdf(gamma_sample(t-1,1),alpha1,delta1)*lognpdf(gamma_sample(t-1,2),alpha2,delta2);
            pi_gamma_hats=lognpdf(gamma1Star,alpha1,delta1)*lognpdf(gamma2Star,alpha2,delta2);
            acceptance_gamma1=(qcurr_gamma1*pi_gamma_hats/qhats_gamma1*pi_gamma);
            acceptance_gamma2=(qcurr_gamma2*pi_gamma_hats/qhats_gamma2*pi_gamma);


            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %Update gamma
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       

            u_gam1=unifrnd(0,1);
            u_gam2=unifrnd(0,1);

            if acceptance_gamma1 >= u_gam1
                gamma_sample(t,1)=gamma1Star;
            else
                gamma_sample(t,1)=gamma_sample(t-1,1);
            end   
            if acceptance_gamma2 >= u_gam2
                gamma_sample(t,2)=gamma2Star;
            else
                gamma_sample(t,2)=gamma_sample(t-1,2);
            end  




        %END OF UPDATES B1, B2, B4 (loop over number of objects);
        end        
    end
    
    
    
    %B3: Updating m;
    %RJMCMC steps
    m_updated_ind=0;
%     latest_m=m_sample(t-1,1);
    [f_n,f_m]=size(f);
    %Determine move type; birth=1/death=2/split=3/merge=4
    if latest_m>1 %p_birth=p_death=p_split=p_merge=0.25;
        p_birth=0.25;
        p_death=0.25;
        p_split=0.25;
        p_merge=0.25;
        move=unifrnd(0,1);
        if move<= p_birth
            move_type=1;
        end
        if (p_birth<move)&&(move<=p_birth+p_death)
            move_type=2;
        end
        if (p_birth+p_death<move)&&(move<=p_birth+p_death+p_split)
            move_type=3;
        end
        if p_birth+p_death+p_split<move
            move_type=4;
        end
    end
    if latest_m==1 %p_birth=p_death=p_split=0.333333;p_merge=0.0;
        p_birth=0.333333;
        p_death=p_birth+0.333333;
        p_split=p_death+0.333333;
        p_merge=0.00;
        move=unifrnd(0,1);
        if move<= p_birth
            move_type=1;
        end
        if (p_birth<move)&&(move<=p_death)
            move_type=2;
        end
        if p_split<move
            move_type=3;
        end
    end
    if latest_m==0 %p_birth=1.0; p_death=p_split=p_merge=0;
        move_type=1;
        p_birth=1.00;
        p_death=0.00;
        p_split=0.00;
        p_merge=0.00;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%% BIRTH MOVE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if move_type==1 %BIRTH MOVE
        %Get current state eta=; mu, sigma
        new_m=latest_m+1;
        cur_mu=muSample(t-1,1:latest_m);
        cur_sigma=sigmaSample(t-1,1:latest_m);
        cur_s=sSample(t-1,1:latest_m);
        %end1=size(cSample);
        cxx=c_xSample(t-1,1:latest_m);
        cyy=c_ySample(t-1,1:latest_m);
        latest_c=[cxx; cyy];
        latest_theta=thetaSample(t-1,1:latest_m);
        latest_gr=g_rSample(t-1,1:latest_m);
        latest_T=TSample(t-1,1:latest_m);

        %Birth move
        
        %Generate random object
        rn_sigma=qsig(mean(NjSample(t-1,1:latest_m))-1,mean(sigmaSample(t-1,1:latest_m)));
        rn_mu=qmu(mean(muSample(t-1,1:latest_m)),rn_sigma);
        rn_s=qs(mean(sSample(t-1,1:latest_m)),mean(sigma_sj0(1,1:latest_m))/10);
        kk=unidrnd(new_m);
        rn_c=qc(latest_c(:,kk),sigma_cj0/10);
        rn_theta=unifrnd(0,pi);
        rn_T=q_T();
        rn_gr=unifrnd(1.005,1.8);

        %Calculate pstar (m+1 and m)
        pi_mu_sigbirth=1/(prod(sigmaSample(t-1,1:latest_m))*rn_sigma);
        pi_mu_sigremain=1/(prod(sigmaSample(t-1,1:latest_m)));
        pi_sbirth=prod(unifpdf(sSample(t-1,1:latest_m),0,sMax))*unifpdf(rn_s,0,sMax);
        pi_sremain=prod(unifpdf(sSample(t-1,1:latest_m),0,sMax));
        s_eta_birth=sum(sSample(t-1,1:latest_m));
        bb1=-gamma_sample(t-1,1)*new_m;
        bb2=gamma_sample(t-1,2)*s_eta_birth;
        pi_cbirth=exp(bb1-bb2);
        pi_cremain=exp(-gamma_sample(t-1,1)*latest_m-gamma_sample(t-1,2)*s_eta_birth);  
        pi_thetabirth=prod(unifpdf(thetaSample(t-1,1:latest_m),0,pi))*unifpdf(rn_theta,0,pi);
        pi_thetaremain=prod(unifpdf(thetaSample(t-1,1:latest_m),0,pi));  
        pi_grbirth=prod(unifpdf(g_rSample(t-1,1:latest_m),1.005,1.8))*unifpdf(rn_gr,1.005,1.8);
        pi_grremain=prod(unifpdf(g_rSample(t-1,1:latest_m),1.005,1.8));
        pi_Tbirth=prod(unidpdf(TSample(t-1,1:latest_m),2))*unidpdf(rn_T,2);
        pi_Tremain=prod(unidpdf(TSample(t-1,1:latest_m),2));
        %No likelihood needed as birth and remain likelihood cancels out

        pstar_birth=pi_mu_sigbirth*pi_sbirth*pi_cbirth*pi_thetabirth*pi_grbirth*pi_Tbirth;
        pstar_remain=pi_mu_sigremain*pi_sremain*pi_cremain*pi_thetaremain*pi_grremain*pi_Tremain;

        pi_birth=(1/rn_sigma)*unifpdf(rn_s,0,sMax)*exp(-gamma_sample(t-1,1)*new_m-gamma_sample(t-1,2)*s_eta_birth)...
            *unifpdf(rn_theta,0,pi)*unifpdf(rn_gr,1.005,1.8)*unidpdf(rn_T,2);
        Jacobian=1;
        r_b1=(pstar_birth*p_death);
        r_b2=(pstar_remain*pi_birth*p_birth);
        if r_b2==0
            r_b2=0.000001;
        end
        r_b=r_b1/r_b2*Jacobian;
        if r_b>1
            r_b=1;
        end
        %Update m
        u_m=unifrnd(0,1);

        if r_b < u_m
            m_sample(t,1)=new_m;
            muSample(t,new_m)=rn_mu;
            sigmaSample(t,new_m)=rn_sigma;
            sSample(t,new_m)=rn_s;
            NjSample(t,new_m)=rn_s;
            c_xSample(t,new_m)=rn_c(1);
            c_ySample(t,new_m)=rn_c(2);
            thetaSample(t,new_m)=rn_theta;
            g_rSample(t,new_m)=rn_gr;
            TSample(t,new_m)=rn_T;
            m_updated_ind=1;
        else
            m_sample(t,1)=m_sample(t-1,1);
            m_updated_ind=1;
        end   
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%% DEATH MOVE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if move_type==2 %DEATH MOVE
        cur_mu=muSample(t,1:latest_m);
        cur_sigma=sigmaSample(t,1:latest_m);
        cur_s=sSample(t,1:latest_m);
        %end1=size(cSample);
        latest_c=[c_xSample(t,1:latest_m); c_ySample(t,1:latest_m)];
        latest_theta=thetaSample(t,1:latest_m);
        latest_gr=g_rSample(t,1:latest_m);
        latest_T=TSample(t,1:latest_m);

        %Choose random object to remove
        kk=unidrnd(latest_m);

        %Death move
        new_m=latest_m-1;
        %Choose random object
        rn_sigma=cur_sigma(1,kk);
        rn_mu=cur_mu(1,kk);
        rn_s=cur_s(1,kk);
        rn_c=latest_c(:,kk);
        rn_theta=latest_theta(1,kk);
        rn_T=latest_T(1,kk);
        rn_gr=latest_gr(1,kk);


        pi_mu_sigdeath=1/(prod(sigmaSample(t,1:latest_m))/rn_sigma);
        pi_mu_sigremain=1/(prod(sigmaSample(t,1:latest_m)));
        pi_sdeath=prod(unifpdf(sSample(t,1:latest_m),0,sMax))/unifpdf(rn_s,0,sMax);
        pi_sremain=prod(unifpdf(sSample(t,1:latest_m),0,sMax));
        s_eta_death=sum(sSample(t,1:latest_m));
        pi_cdeath=exp(-gamma_sample(t,1)*new_m-gamma_sample(t,2)*s_eta_death);
        pi_cremain=exp(-gamma_sample(t,1)*latest_m-gamma_sample(t,2)*s_eta_death);  
        pi_thetadeath=prod(unifpdf(thetaSample(t,1:latest_m),0,pi))/unifpdf(rn_theta,0,pi);
        pi_thetaremain=prod(unifpdf(thetaSample(t,1:latest_m),0,pi));  
        pi_grdeath=prod(unifpdf(g_rSample(t,1:latest_m),1.005,1.8))/unifpdf(rn_gr,1.005,1.8);
        pi_grremain=prod(unifpdf(g_rSample(t,1:latest_m),1.005,1.8));
        pi_Tdeath=prod(unidpdf(TSample(t,1:latest_m),2))/unidpdf(rn_T,2);
        pi_Tremain=prod(unidpdf(TSample(t,1:latest_m),2));
        %No likelihood needed as death and remain likelihood cancels out

        pstar_death=pi_mu_sigdeath*pi_sdeath*pi_cdeath*pi_thetadeath*pi_grdeath*pi_Tdeath;
        pstar_remain=pi_mu_sigremain*pi_sremain*pi_cremain*pi_thetaremain*pi_grremain*pi_Tremain;

        pi_death=(1/rn_sigma)*unifpdf(rn_s,0,sMax)*exp(-gamma_sample(t,1)*new_m-gamma_sample(t,2)*s_eta_death)...
            *unifpdf(rn_theta,0,pi)*unifpdf(rn_gr,1.005,1.8)*unidpdf(rn_T,2);
        Jacobian=1;
        r_b1=(pstar_death*p_death);
        r_b2=(pstar_remain*pi_death*p_death);
        if r_b2==0
            r_b2=0.000001;
        end
        r_b=r_b1/r_b2*Jacobian;
        if r_b==0
            r_b=0.000001;
        end
        if r_b>1
            r_b=1;
        end
        r_d=1/r_b;
        
        %Update m
        u_m=unifrnd(0,1);

        if r_d < u_m
            m_sample(t,1)=new_m;

            muSample(t,kk:new_m)=muSample(t,kk+1:latest_m);
            muSample(t,latest_m)=0;

            sigmaSample(t,kk:new_m)=sigmaSample(t,kk+1:latest_m);
            sigmaSample(t,latest_m)=0;

            sSample(t,kk:new_m)=sSample(t,kk+1:latest_m);
            sSample(t,latest_m)=0;

            NjSample(t,kk:new_m)=NjSample(t,kk+1:latest_m);
            NjSample(t,latest_m)=0;

            c_xSample(t,kk:new_m)=c_xSample(t,kk+1:latest_m);
            c_xSample(t,latest_m)=0;

            c_ySample(t,kk:new_m)=c_ySample(t,kk+1:latest_m);
            c_ySample(t,latest_m)=0;

            g_rSample(t,kk:new_m)=g_rSample(t,kk+1:latest_m);
            g_rSample(t,latest_m)=0;

            thetaSample(t,kk:new_m)=thetaSample(t,kk+1:latest_m);
            thetaSample(t,latest_m)=0;

            TSample(t,kk:new_m)=TSample(t,kk+1:latest_m);
            TSample(t,latest_m)=0;
            m_updated_ind=1;
        else
            m_sample(t,1)=m_sample(t-1,1);
            m_updated_ind=1;
        end   
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%% MERGE MOVE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

    if move_type==4 %MERGE MOVE

        cur_mu=muSample(t,1:latest_m);
        cur_sigma=sigmaSample(t,1:latest_m);
        cur_s=sSample(t,1:latest_m);
        %end1=size(cSample);
        latest_c=[c_xSample(t,1:latest_m); c_ySample(t,1:latest_m)];
        latest_theta=thetaSample(t,1:latest_m);
        latest_gr=g_rSample(t,1:latest_m);
        latest_T=TSample(t,1:latest_m);

        %Choose random object to merge
        kk=unidrnd(latest_m);

        %Merge move
        new_m=latest_m-1;

        %Generate merged object


        if kk < latest_m
            kk1=kk+1;
        else
            kk1=kk-1;
        end

       si=cur_s(1,kk);
       sj=cur_s(1,kk1);
       xi=latest_c(1,kk);
       xj=latest_c(1,kk1);
       yi=latest_c(2,kk);
       yj=latest_c(2,kk1);
       rn_s=sqrt(si+sj);
       rn_cx=(si*xi+sj*xj)/(si+sj);
       rn_cy=(si*yi+sj*yj)/(si+sj);
       rn_c=[rn_cx;rn_cy];
       rn_kk=unifrnd(0,1);
       if rn_kk<0.5
           kk2=kk;
       else
           kk2=kk1;
       end
       rn_theta=latest_theta(kk2);
       rn_T=latest_T(kk2);
       rn_gr=latest_gr(kk2);
       rn_sigma=cur_sigma(kk2);
       rn_mu=cur_mu(kk2);  
       u1=sqrt((xi-xj)^2+(yi-yj)^2);
       u2=atan((yj-yi)/(u1));
       u3=(si^2-sj^2)/(si^2+sj^2);
       u4=latest_theta(kk1);
       u5=latest_T(kk1);
       u6=latest_gr(kk1);

       pi_mu_sigmerge=1/(prod(sigmaSample(t,1:latest_m))*rn_sigma/(sigmaSample(t,kk)*sigmaSample(t,kk1)));
       pi_mu_sigremain=1/(prod(sigmaSample(t,1:latest_m)));
       pi_smerge=prod(unifpdf(sSample(t,1:latest_m),0,sMax))*unifpdf(rn_s,0,sMax)...
           /(unifpdf(sSample(t,kk),0,sMax)*unifpdf(sSample(t,kk1),0,sMax));
       pi_sremain=prod(unifpdf(sSample(t,1:latest_m),0,sMax));
       s_eta_merge=sum(sSample(t,1:latest_m));
       pi_cmerge=exp(-gamma_sample(t,1)*new_m-gamma_sample(t,2)*s_eta_merge);
       pi_cremain=exp(-gamma_sample(t,1)*latest_m-gamma_sample(t,2)*s_eta_merge);  
       pi_thetamerge=prod(unifpdf(thetaSample(t,1:latest_m),0,pi))*unifpdf(rn_theta,0,pi)...
           /(unifpdf(thetaSample(t,kk),0,pi)*unifpdf(thetaSample(t,kk1),0,pi));
       pi_thetaremain=prod(unifpdf(thetaSample(t,1:latest_m),0,pi));  
       pi_grmerge=prod(unifpdf(g_rSample(t,1:latest_m),1.005,1.8))*unifpdf(rn_gr,1.005,1.8)...
           /(unifpdf(g_rSample(t,kk),1.005,1.8)*unifpdf(g_rSample(t,kk1),1.005,1.8));
       pi_grremain=prod(unifpdf(g_rSample(t,1:latest_m),1.005,1.8));
       pi_Tmerge=prod(unidpdf(TSample(t,1:latest_m),2))*unidpdf(rn_T,2)...
           /(unidpdf(latest_T(kk),2)*unidpdf(latest_T(kk1),2));
       pi_Tremain=prod(unidpdf(TSample(t,1:latest_m),2));
        %No likelihood needed as death and remain likelihood cancels out

        pstar_merge=pi_mu_sigmerge*pi_smerge*pi_cmerge*pi_thetamerge*pi_grmerge*pi_Tmerge;
        pstar_remain=pi_mu_sigremain*pi_sremain*pi_cremain*pi_thetaremain*pi_grremain*pi_Tremain;

        pi_merge=unifpdf(u1,0,sMax)*unifpdf(u2,1.005,1.8)*unifpdf(u3,-1,1)...
            *unifpdf(u4,1.005,1.8)*unidpdf(u5,2)*unifpdf(u6,1.005,1.8);
        Jacobian=1;
        r_m1=(pstar_merge*p_split*pi_merge);
        r_m2=(pstar_remain*p_merge);
        if r_m2==0
            r_m2=0.00001;
        end
        r_m=(r_m1/r_m2)*Jacobian;
        if r_m>1
            r_m=1;
        end
        %Update m
        u_m=unifrnd(0,1);

        if r_m < u_m && kk<latest_m
            m_sample(t,1)=new_m;

            muSample(t,kk:latest_m-2)=muSample(t,kk+2:latest_m);
            muSample(t,latest_m-1)=rn_mu;
            muSample(t,latest_m)=0;

            sigmaSample(t,kk:latest_m-2)=sigmaSample(t,kk+2:latest_m);
            sigmaSample(t,latest_m-1)=rn_sigma;
            sigmaSample(t,latest_m)=0;

            sSample(t,kk:latest_m-2)=sSample(t,kk+2:latest_m);
            sSample(t,latest_m-1)=rn_s;
            sSample(t,latest_m)=0;

            NjSample(t,kk:latest_m-2)=NjSample(t,kk+2:latest_m);
            NjSample(t,latest_m-1)=rn_s;
            NjSample(t,latest_m)=0;

            c_xSample(t,kk:latest_m-2)=c_xSample(t,kk+2:latest_m);
            c_xSample(t,latest_m-1)=rn_cx;
            c_xSample(t,latest_m)=0;

            c_ySample(t,kk:latest_m-2)=c_ySample(t,kk+2:latest_m);
            c_ySample(t,latest_m-1)=rn_cy;
            c_ySample(t,latest_m)=0;

            g_rSample(t,kk:latest_m-2)=g_rSample(t,kk+2:latest_m);
            g_rSample(t,latest_m-1)=rn_gr;
            g_rSample(t,latest_m)=0;

            thetaSample(t,kk:latest_m-2)=thetaSample(t,kk+2:latest_m);
            thetaSample(t,latest_m-1)=rn_theta;
            thetaSample(t,latest_m)=0;

            TSample(t,kk:latest_m-2)=TSample(t,kk+2:latest_m);
            TSample(t,latest_m-1)=rn_T;
            TSample(t,latest_m)=0;
            m_updated_ind=1;
        end   
        if r_m < u_m && kk>=latest_m
            m_sample(t,1)=new_m;

            muSample(t,kk-1:latest_m-2)=muSample(t,kk+1:latest_m);
            muSample(t,latest_m-1)=rn_mu;
            muSample(t,latest_m)=0;

            sigmaSample(t,kk-1:latest_m-2)=sigmaSample(t,kk+1:latest_m);
            sigmaSample(t,latest_m-1)=rn_sigma;
            sigmaSample(t,latest_m)=0;

            sSample(t,kk-1:latest_m-2)=sSample(t,kk+1:latest_m);
            sSample(t,latest_m-1)=rn_s;
            sSample(t,latest_m)=0;

            NjSample(t,kk-1:latest_m-2)=NjSample(t,kk+1:latest_m);
            NjSample(t,latest_m-1)=rn_s;
            NjSample(t,latest_m)=0;

            c_xSample(t,kk-1:latest_m-2)=c_xSample(t,kk+1:latest_m);
            c_xSample(t,latest_m-1)=rn_cx;
            c_xSample(t,latest_m)=0;

            c_ySample(t,kk-1:latest_m-2)=c_ySample(t,kk+1:latest_m);
            c_ySample(t,latest_m-1)=rn_cy;
            c_ySample(t,latest_m)=0;

            g_rSample(t,kk-1:latest_m-2)=g_rSample(t,kk+1:latest_m);
            g_rSample(t,latest_m-1)=rn_gr;
            g_rSample(t,latest_m)=0;

            thetaSample(t,kk-1:latest_m-2)=thetaSample(t,kk+1:latest_m);
            thetaSample(t,latest_m-1)=rn_theta;
            thetaSample(t,latest_m)=0;

            TSample(t,kk-1:latest_m-2)=TSample(t,kk+1:latest_m);
            TSample(t,latest_m-1)=rn_T;
            TSample(t,latest_m)=0;
            m_updated_ind=1;
        end
        if r_m >= u_m
            m_sample(t,1)=m_sample(t-1,1);
            m_updated_ind=1;
        end   

    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%% SPLIT MOVE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     

    if move_type==3 %SPLIT MOVE

        cur_mu=muSample(t,1:latest_m);
        cur_sigma=sigmaSample(t,1:latest_m);
        cur_s=sSample(t,1:latest_m);
        %end1=size(cSample);
        latest_c=[c_xSample(t,1:latest_m); c_ySample(t,1:latest_m)];
        latest_theta=thetaSample(t,1:latest_m);
        latest_gr=g_rSample(t,1:latest_m);
        latest_T=TSample(t,1:latest_m);

        %Choose random object to SPLIT
        kk=unidrnd(latest_m);

        %Split move
        new_m=latest_m+1;

        %Generate split object

        if kk < latest_m
            kk1=kk+1;
        else
            kk1=kk-1;
        end

       si=cur_s(1,kk);
       sj=cur_s(1,kk1);
       xi=latest_c(1,kk);
       xj=latest_c(1,kk1);
       yi=latest_c(2,kk);
       yj=latest_c(2,kk1);
       rn_s=sqrt(si+sj);
       rn_cx=(si*xi+sj*xj)/(si+sj);
       rn_cy=(si*yi+sj*yj)/(si+sj);
       rn_c=[rn_cx;rn_cy];
       rn_kk=unifrnd(0,1);
       if rn_kk<0.5
           kk2=kk;
       else
           kk2=kk1;
       end
       rn_theta=latest_theta(kk2);
       rn_T=latest_T(kk2);
       rn_gr=latest_gr(kk2);
       rn_sigma=cur_sigma(kk2);
       rn_mu=cur_mu(kk2);  
       u1=sqrt((xi-xj)^2+(yi-yj)^2);
       u2=atan((yj-yi)/(u1));
       u3=(si^2-sj^2)/(si^2+sj^2);
       u4=latest_theta(kk1);
       u5=latest_T(kk1);
       u6=latest_gr(kk1);

       pi_mu_sigmerge=1/(prod(sigmaSample(t,1:latest_m))*rn_sigma/(sigmaSample(t,kk)*sigmaSample(t,kk1)));
       pi_mu_sigremain=1/(prod(sigmaSample(t,1:latest_m)));
       pi_smerge=prod(unifpdf(sSample(t,1:latest_m),0,sMax))*unifpdf(rn_s,0,sMax)...
           /(unifpdf(sSample(t,kk),0,sMax)*unifpdf(sSample(t,kk1),0,sMax));
       pi_sremain=prod(unifpdf(sSample(t,1:latest_m),0,sMax));
       s_eta_merge=sum(sSample(t,1:latest_m));
       pi_cmerge=exp(-gamma_sample(t,1)*new_m-gamma_sample(t,2)*s_eta_merge);
       pi_cremain=exp(-gamma_sample(t,1)*latest_m-gamma_sample(t,2)*s_eta_merge);  
       pi_thetamerge=prod(unifpdf(thetaSample(t,1:latest_m),0,pi))*unifpdf(rn_theta,0,pi)...
           /(unifpdf(thetaSample(t,kk),0,pi)*unifpdf(thetaSample(t,kk1),0,pi));
       pi_thetaremain=prod(unifpdf(thetaSample(t,1:latest_m),0,pi));  
       pi_grmerge=prod(unifpdf(g_rSample(t,1:latest_m),1.005,1.8))*unifpdf(rn_gr,1.005,1.8)...
           /(unifpdf(g_rSample(t,kk),1.005,1.8)*unifpdf(g_rSample(t,kk1),1.005,1.8));
       pi_grremain=prod(unifpdf(g_rSample(t,1:latest_m),1.005,1.8));
       pi_Tmerge=prod(unidpdf(TSample(t,1:latest_m),2))*unidpdf(rn_T,2)...
           /(unidpdf(latest_T(kk),2)*unidpdf(latest_T(kk1),2));
       pi_Tremain=prod(unidpdf(TSample(t,1:latest_m),2));
        %No likelihood needed as death and remain likelihood cancels out

        pstar_merge=pi_mu_sigmerge*pi_smerge*pi_cmerge*pi_thetamerge*pi_grmerge*pi_Tmerge;
        pstar_remain=pi_mu_sigremain*pi_sremain*pi_cremain*pi_thetaremain*pi_grremain*pi_Tremain;

        pi_merge=unifpdf(u1,0,sMax)*unifpdf(u2,1.005,1.8)*unifpdf(u3,-1,1)...
            *unifpdf(u4,1.005,1.8)*unidpdf(u5,2)*unifpdf(u6,1.005,1.8);
        Jacobian=1;
        r_m1=(pstar_merge*p_split*pi_merge);
        r_m2=(pstar_remain*p_merge);
        if r_m2==0
            r_m2=0.000001;
        end
        r_m=r_m1/r_m2*Jacobian;
        if r_m==0
            r_m=0.000001;
        end
        if r_m>1
            r_m=1;
        end
        r_s=1/r_m;
        if r_s>1
            r_s=1;
        end
        %Update m
        u_m=unifrnd(0,1);

        if r_s < u_m
            m_sample(t,1)=new_m;

            muSample(t,kk+1)=rn_mu;
            muSample(t,kk+2:new_m)=muSample(t,kk+1:latest_m);

            sigmaSample(t,kk+1)=rn_sigma;
            sigmaSample(t,kk+2:new_m)=sigmaSample(t,kk+1:latest_m);

            sSample(t,kk+1)=rn_s;
            sSample(t,kk+2:new_m)=sSample(t,kk+1:latest_m);

            NjSample(t,kk+1)=rn_s;
            NjSample(t,kk+2:new_m)=NjSample(t,kk+1:latest_m);

            c_xSample(t,kk+1)=rn_cx;
            c_xSample(t,kk+2:new_m)=c_xSample(t,kk+1:latest_m);

            c_ySample(t,kk+1)=rn_cy;
            c_ySample(t,kk+2:new_m)=c_ySample(t,kk+1:latest_m);

            g_rSample(t,kk+1)=rn_gr;
            g_rSample(t,kk+2:new_m)=g_rSample(t,kk+1:latest_m);

            thetaSample(t,kk+1)=rn_theta;
            thetaSample(t,kk+2:new_m)=thetaSample(t,kk+1:latest_m);

            TSample(t,kk+1)=rn_T;
            TSample(t,kk+2:new_m)=TSample(t,kk+1:latest_m);
            m_updated_ind=1;
        end   

        if r_s >= u_m
            m_sample(t,1)=m_sample(t-1,1);
            m_updated_ind=1;
        end   

    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %STEP A2: DRAW AUXILIARY VARIABLES FROM THE LIKELIHOOD FUNCTION
    K = m_sample(t,1)*8+2+1; %8 parameters: mu, sigma, s, cx, cy, theta, g_r, T + gamma1, gamma2 + m
    mm=m_sample(t,1);
    latest_parms=[m_sample(t,1), muSample(t,1:mm),sigmaSample(t,1:mm)...
        ,sSample(t,1:mm),c_xSample(t,1:mm),c_ySample(t,1:mm)...
        ,thetaSample(t,1:mm),g_rSample(t,1:mm),TSample(t,1:mm)...
        ,gamma_sample(t,1:2)];
    auxil_parms=normrnd(mean(muSample(t,1:mm)),mean(sigmaSample(t,1:mm)),1,K);
    
    %STEP A3: COMPUTE MCMH ACCEPTANCE RATIO
    f_zStar=normpdf(auxil_parms,mean(muSample(t,1:mm)),mean(sigmaSample(t,1:mm)));
    f_zCur=normpdf(latest_parms,mean(muSample(t,1:mm)),mean(sigmaSample(t,1:mm)));

    ratio_z=zeros(1,K);
    for i= 1 : K
        ratio_z(1,i)=f_zStar(1,i)/f_zCur(1,i);
    end
    R_hat=(sum(ratio_z,2))/K;
    if R_hat==0
        R_hat=0.000001;
    end
    if R_hat>1
        R_hat=1;
    end
    r_hat=1/R_hat;
    
    %STEP A4: ACCEPT/REJECT PROPOSAL
    u_z=unifrnd(0,1);

    if r_hat < u_z
        %Reject proposal
        m_sample(t,:)=m_sample(t-1,:);
        muSample(t,:)=muSample(t-1,:);
        sigmaSample(t,:)=sigmaSample(t-1,:);
        sSample(t,:)=sSample(t-1,:);
        c_xSample(t,:)=c_xSample(t-1,:);
        c_ySample(t,:)=c_ySample(t-1,:);
        thetaSample(t,:)=thetaSample(t-1,:);
        g_rSample(t,:)=g_rSample(t-1,:);
        TSample(t,:)=TSample(t-1,:);
        gamma_sample(t,:)=gamma_sample(t-1,:);
    end

    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    
    
end
% Final parameters:
% nSamples=t;
nSamples=t-1;
final_m0=m_sample(1:nSamples,1);
final_m=m_sample(nSamples,1);
final_mu=muSample(1:nSamples,1:final_m);
final_sigma=sigmaSample(1:nSamples,1:final_m);
final_s=sSample(1:nSamples,1:final_m);
final_cx=c_xSample(1:nSamples,1:final_m);
final_cy=c_ySample(1:nSamples,1:final_m);
final_theta=thetaSample(1:nSamples,1:final_m);
final_gr=g_rSample(1:nSamples,1:final_m);
final_T=TSample(1:nSamples,1:final_m);
final_gamma=gamma_sample(1:nSamples,:);

% final_m=m_sample(t,1);
% final_mu=muSample(t,1:final_m);
% final_sigma=sigmaSample(t,1:final_m);
% final_s=sSample(t,1:final_m);
% final_cx=c_xSample(t,1:final_m);
% final_cy=c_ySample(t,1:final_m);
% final_theta=thetaSample(t,1:final_m);
% final_gr=g_rSample(t,1:final_m);
% final_T=TSample(t,1:final_m);
% final_gamma=gamma_sample(t,:);

%Create simulated image

% input ellipse parameters
n=90;
theta_grid = zeros(1,n);
for k = 1: n
        theta_grid(k) = 2 *pi*(k - 1)/n;
end

phi = transpose(final_theta(nSamples,1:final_m));
X0=transpose(final_cx(nSamples,1:final_m));
Y0=transpose(final_cy(nSamples,1:final_m));
S0=transpose(final_s(nSamples,1:final_m));
a=transpose(final_gr(nSamples,1:final_m));
b=transpose(final_gr(nSamples,1:final_m));
shape_x_r=zeros(final_m,n);
shape_y_r=zeros(final_m,n);


for k=1:final_m
    b(k)=1/a(k);
    shape_x_r(k,:)  = X0(k) + a(k)*cos( theta_grid );
    shape_y_r(k,:)  = Y0(k) + b(k)*sin( theta_grid );
    % the ellipse in x and y coordinates 
    R = [ cos(phi(k)) sin(phi(k)); -sin(phi(k)) cos(phi(k)) ];

    %Define a rotation matrix
    %let's rotate the ellipse to some angle phii
    rotated_shape = R * [shape_x_r(k,:);shape_y_r(k,:)];
    plot(rotated_shape(1,:),-1*rotated_shape(2,:),'.');
%     title('Location of simulated objects')
    set(gca,'YTick',[])
    set(gca,'YTickLAbel',[])
    set(gca,'XTick',[])
    set(gca,'XTickLAbel',[])
    hold on
    
 
end
 xlabel('Object location: x-coordinate value')
 ylabel('Object location: y-coordinate value')
 imshow(f)
 hold off
 
 plot(final_m0)
 ylim([min(final_m0)-5,max(final_m0)+5])
%  title('Number of simulated objects over time')
 xlabel('Number of simulations')
 ylabel('Number of objects')
 
 %Gamma plots
 %plot(gamma_sample(1:nSamples,1))
 %ylim([min(final_m0)-5,max(final_m0)+5])
%  title('Number of simulated objects over time')
 %xlabel('Number of simulations')
% ylabel('Simulated value for Gamma 1')
 
 % Distribution of number of objects
 m0_dist = final_m0(501:nSamples,1);
 
 histogram(final_m0(1:nSamples,1));
 xlabel('Number of objects identified')
 ylabel('Number of observations in sample')
 sample_mode=mode(final_m0);
 
    kk1=11;
    kk2=15;
    
    plot(final_mu(:,kk1:kk2))
    ylim([min(min(final_mu(:,kk1:kk2)))-5,max(max(final_mu(:,kk1:kk2)))+5])
%     title(['Object ' num2str(kk) ' mean over time'])
    xlabel('Number of simulations')
    ylabel('Mean value')

    plot(final_sigma(:,kk1:kk2))
    ylim([min(min(final_sigma(:,kk1:kk2)))-5,max(max(final_sigma(:,kk1:kk2)))+5])
%     title(['Object ' num2str(kk1:kk2) ' variance over time'])
    xlabel('Number of simulations')
    ylabel('Variance value')
    
    plot(final_theta(:,kk1:kk2))
    ylim([min(min(final_theta(:,kk1:kk2)))-0.5,max(max(final_theta(:,kk1:kk2)))+0.5])
%     title(['Object ' num2strkk1:kk2kk) ' rotation over time'])
    xlabel('Number of simulations')
    ylabel('Rotation (in radians)')
    
    plot(final_T(:,kk1:kk2))
    ylim([min(min(final_T(:,kk1:kk2)))-0.5,max(max(final_T(:,kk1:kk2)))+0.5])
%     title(['Object ' num2str(kk1:kk2) ' template over time'])
    xlabel('Number of simulations')
    ylabel('Object template')
    legend('1: Elipse  2: Circle')
