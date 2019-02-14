import numpy as np
import pandas as pd
import scipy as sc


log = np.log
gamma = sc.special.gamma
digamma = sc.special.digamma
det = np.linalg.det
diag = np.diag
outer = np.outer
tr = np.trace
dot =  np.dot
concat = np.concatenate
normal = np.random.normal
arr = np.array
cstack = np.column_stack
sqrt = np.sqrt
inv = np.linalg.inv
norm = sc.stats.norm
append = np.append
slogdet = np.linalg.slogdet



class BEMKL:
    def __init__(self, max_iter=100, sparse=False):
        self.mu_a = None
        self.cov_a = None
        self.mu_b_e = None
        self.cov_b_e = None
        self.v = 1.
        self.max_iter = max_iter
        self.sparse = sparse


    def bemkl_binary(self, K, y):
        np.random.seed(123)

        MAX_ITER = self.max_iter
        P = K.shape[0]
        N = K.shape[1]

        # Hyper Parameters (prior)
        alpha_lambda, beta_lambda = 1.0,1.0
        alpha_gamma, beta_gamma = 1.0,1.0
        alpha_omega, beta_omega = 1.0,1.0

        if self.sparse:
            alpha_omega, beta_omega = 10.0**-10, 10.0**10

        v = self.v

        # The conditionals parameters
        p_alpha_lambda = np.repeat(alpha_lambda, N)
        p_beta_lambda = np.repeat(beta_lambda, N)
        p_alpha_gamma = alpha_gamma
        p_beta_gamma = beta_gamma
        p_alpha_omega = np.repeat(alpha_omega, P)
        p_beta_omega = np.repeat(beta_omega, P)


        mu_b = 0
        var_b = 1
        mu_e = np.ones(P)
        mu_b_e = np.concatenate(([mu_b],mu_e))
        cov_e = np.eye(P)
        cov_b_e = np.zeros((P+1,P+1))
        cov_b_e[0,0] = var_b
        cov_b_e[1:,1:] = cov_e
        corr_e = cov_b_e[1:,1:] + outer(mu_b_e[1:], mu_b_e[1:])
        print 'COV_b_e', cov_b_e

        mu_a = np.random.normal(0,1, size=N)
        cov_a = diag(np.repeat((alpha_lambda*beta_lambda)**-1, N))
        corr_a = cov_a + outer(mu_a, mu_a)

        print 'cov_a',cov_a[:5,:5]

        mu_g = (np.abs(np.random.normal(0,1,size=(P,N))) + v) * y

        print 'mu_g', mu_g[:,:3], mu_g[:,-3:]


        # inter-column covariance: NxPxP
        cov_g = np.zeros((N,P,P))
        cov_g[:,:,:] = np.eye(P)
        corr_g = cov_g + arr([outer(mu_g[:,i], mu_g[:,i]) for i in range(N)])

        mu_f = (np.random.rand(N) + 1e-5) * y

        print 'mu_f',mu_f[:3], mu_f[-3:]

        E_a2 = np.diag(cov_a) + mu_a**2
        E_b2 = alpha_gamma * beta_gamma + mu_b**2
        E_e2 = np.diag(cov_b_e[1:,1:]) + mu_b_e[1:]**2
        E_be = 0


        thresh = 1.0 * 10**-4
        ELBO_init = 0

        K2 = np.sum([dot(K[i,...], K[i,...].T) for i in range(P)], axis=0)

        ######################################################################

        it = 1
        v = 1
        trace = False

        while True:
            # Update the parameters:
            ########################

            '''
            ############
            Update a
            ############
            '''
            p_alpha_lambda = np.repeat(alpha_lambda + 0.5, N)
            p_beta_lambda = (beta_lambda**-1 + E_a2/2.0)**-1
            E_lambda = p_alpha_lambda * p_beta_lambda


            cov_a = inv(diag(E_lambda) + K2)
            z = np.sum([dot(K[i,:,:], mu_g[i,:]) for i in range(P)], axis=0)
            mu_a = dot(cov_a, z)
            corr_a = cov_a + outer(mu_a, mu_a)
            E_a2 = diag(cov_a) + mu_a**2

            print 'Updated a...'
            print 'mu_a', mu_a[:5]
            print 'cov_a', cov_a[:5,:5]

            '''
            ############
            Update G
            ############
            '''
            cov_g[:,:,:] = inv(np.eye(P) + corr_e)
            mu_g = cstack([dot(cov_g[i,:,:], (dot(K[:,i,:], mu_a) + mu_f[i]* mu_b_e[1:]
                                              - E_be)) for i in range(N)])
            corr_g = cov_g + arr([outer(mu_g[:,i], mu_g[:,i]) for i in range(N)])
            corr_G = dot(mu_g,mu_g.T) + cov_g.sum(axis=0)  # exactly the same
            cov_G = cov_g.sum(axis=0)

            print 'Updated g...'
            print 'mu_g',mu_g[:,:3]
            print mu_g[:,-3:]
            print 'cov_g', cov_g[:1,...]
            print 'corr_G', corr_G

            '''
            ############
            Update e, b
            ############
            '''
            p_alpha_gamma = alpha_gamma + 0.5
            p_beta_gamma = (beta_gamma**-1 + E_b2/2.0)**-1

            p_alpha_omega = np.repeat(alpha_omega + 0.5, P)
            p_beta_omega = (beta_omega**-1 + E_e2/2.0)**-1

            E_gamma = p_alpha_gamma * p_beta_gamma
            E_omega = p_alpha_omega * p_beta_omega


            cov_b_e[0,0] = E_gamma + N
            cov_b_e[1:,0] = mu_g.sum(axis=1)
            cov_b_e[0,1:] = cov_b_e[1:,0]
            cov_b_e[1:,1:] = diag(E_omega) + corr_G
            try:  cov_b_e = inv(cov_b_e)
            except: cov_b_e = inv(cov_b_e + 1e-5 * np.eye((P+1,P+1)))

            z = concat(([mu_f.sum()], dot(mu_g,mu_f).squeeze()))
            mu_b_e = dot(cov_b_e, z)

            E_be = mu_b_e[0] * mu_b_e[1:] + cov_b_e[0,1:]
            E_b2 = cov_b_e[0,0] + mu_b_e[0]**2
            E_e2 = diag(cov_b_e[1:,1:]) + mu_b_e[1:]**2
            corr_e = cov_b_e[1:,1:] + outer(mu_b_e[1:], mu_b_e[1:])

            print 'Updated e:'
            print 'mu_b_e', mu_b_e
            print 'cov_b_e', cov_b_e
            print 'corr_e', corr_e
            print 'E_be', E_be

            '''
            ############
            Update f
            ############
            '''
            LU = arr(map(lambda x: (-1e40, -v) if x==-1 else (v,1e40), y))
            eg = dot(mu_b_e[1:].T, mu_g)
            alpha_f = LU[:,0] - eg - mu_b_e[0]
            beta_f = LU[:,1] - eg - mu_b_e[0]
            pdf_alpha_f = norm.pdf(alpha_f)
            pdf_beta_f = norm.pdf(beta_f)
            Z = norm.cdf(beta_f) - norm.cdf(alpha_f)
            msk = (Z == 0.0)
            Z[msk] = 1.0
            mu_f = eg + mu_b_e[0] + (pdf_alpha_f - pdf_beta_f)/Z
            var_f = 1. + (alpha_f * pdf_alpha_f - beta_f * pdf_beta_f)/Z - (pdf_alpha_f - pdf_beta_f)**2/Z**2
            E_f2 = var_f + mu_f**2

            print 'Updated f'
            print 'mu_f',mu_f[:3],mu_f[-3:]
            print 'var_f',var_f[:3]
            print 'E_f2', E_f2[:3]
            #print Z[:10],pdf_alpha_f[:3], pdf_beta_f[:3]

            '''
            ###################
            Calculate the ELBO
            ###################
            '''
            if trace:
                E_log_lambda = digamma(p_alpha_lambda) + log(p_beta_lambda)
                E_lp_lambda = ((alpha_lambda - 1) * E_log_lambda - 1.0/beta_lambda * E_lambda - log(gamma(alpha_lambda))
                        - alpha_lambda * log(beta_lambda)).sum(axis=0)
                print 'lp_lam', E_lp_lambda

                # E(log(p(a|lambda)))
                E_lp_a_lambda = ( - 0.05 * tr(dot(diag(E_lambda), corr_a)).sum() - 0.5 * N *log(2.0*np.pi)
                                 + 0.05 * log(np.prod(E_lambda)))
                print 'lp_a_lam', E_lp_a_lambda

                # E(log(p(G|a,K)))
                E_g2 = arr([(diag(cov_g[i,...]) + mu_g[:,i]**2).sum() for i in range(N)])
                kga = arr([dot(K[:,:,i].T , outer(mu_g[:,i],mu_a)) for i in range(N)])

                kaa = arr([dot(K[:,:,i].T , dot(K[:,:,i],corr_a)) for i in range(N)])
                #print 'kaa', kaa[0,:5,:5]

                E_lp_G = np.sum([-.5 * E_g2[i] + tr(kga[i,...]) - 0.5 * tr(kaa[i,...])
                                 - 0.5 * P *log(2.0*np.pi) for i in range(N)])
                print 'lp_g', E_lp_G

                # E(log(p(gamma)))
                E_log_gamma = digamma(p_alpha_gamma) + log(p_beta_gamma)
                print 'E_log_gamma', E_log_gamma
                E_lp_gamma = ((alpha_gamma - 1) * E_log_gamma - E_gamma/beta_gamma
                              - log(gamma(alpha_gamma)) - alpha_gamma * log(beta_gamma))
                print 'lp_gamma', E_lp_gamma

                # E(log(p(b|gamma)))
                E_lp_b = (- 0.5 * E_gamma * E_b2 - 0.5 *log(2.0*np.pi) + 0.5 * log(E_gamma))
                print 'lp_b', E_lp_b

                # E(log(p(omega)))
                E_log_omega = digamma(p_alpha_omega) + log(p_beta_omega)
                E_lp_omega = ((alpha_omega - 1) * E_log_omega - E_omega/beta_omega - log(gamma(alpha_omega))
                              - alpha_omega * log(beta_omega) ).sum()
                print 'lp_omega', E_lp_omega

                # E(log(p(e|omega)))
                E_lp_e = ( -0.5 * tr(dot(diag(E_omega), corr_e)) - 0.5 * P * log(2.0*np.pi)
                         + 0.05 * log(np.prod(E_omega)))
                print 'lp_e', E_lp_e

                # E(log(p(f|e,b,G)))
                z1 = (- 0.5 * E_f2).sum()
                z2 = ((dot(mu_b_e[1:], mu_g) + mu_b_e[0]) * mu_f ).sum()
                z3 = - 0.5 * np.sum(dot(corr_e, corr_G)) - 0.5 * (dot(E_be, mu_g) + E_b2).sum()

                E_lp_f =  z1 + z2 + z3 - 0.5 *  N *log(2.0*np.pi)
                print 'lp_f'
                print E_lp_f


                # E(log(q(lambda)))
                E_lq_lambda = (-p_alpha_lambda - log(p_beta_lambda) - log(gamma(p_alpha_lambda))
                               - (1-p_alpha_lambda) * digamma(p_alpha_lambda)).sum()
                print 'lq_lam', E_lq_lambda

                # E(log(q(a)))
                E_lq_a = -0.5 * N * (log(2.0*np.pi)+1) - 0.5 * slogdet(cov_a)[1]
                print 'lq_a', E_lq_a

                # E(log(q(G)))
                #E_lq_G = (-0.5 * P *(log(2.0*np.pi)+1) - 0.5 * np.array([slogdet(cov_g[i])[0] for i in range(N)])).sum()
                E_lq_G = -0.5 * N * P *(log(2.0*np.pi)+1) - 0.5 * slogdet(cov_G)[1]
                print 'lq_g', E_lq_G

                # E(log(q(gamma)))
                E_lq_gamma = (-p_alpha_gamma - log(p_beta_gamma) - log(gamma(p_alpha_gamma))
                               - (1-p_alpha_gamma) * digamma(p_alpha_gamma))
                print 'lq_gamma', E_lq_gamma

                # E(log(q(omega)))
                E_lq_omega = (-p_alpha_omega - log(p_beta_omega) - log(gamma(p_alpha_omega))
                               - (1-p_alpha_omega) * digamma(p_alpha_omega)).sum()
                print 'lq_omega', E_lq_omega

                # E(log(q(b,e)))
                E_lq_b_e = (0.5 * (P+1) *(log(2.0*np.pi)+1) + 0.5 * slogdet(cov_b_e)[1])
                print 'lq_be', E_lq_b_e

                # E(log(q(f)))
                E_lq_f = (-0.5 * (log(2.0*np.pi) + var_f) - log(Z)).sum()
                print 'lq_f', E_lq_f


                ELBO = (E_lp_lambda + E_lp_a_lambda + E_lp_G + E_lp_gamma + E_lp_b + E_lp_omega + E_lp_e + E_lp_f
                        - E_lq_lambda - E_lq_a - E_lq_G - E_lq_gamma - E_lq_omega - E_lq_b_e - E_lq_f)

                print "ITER %d: %f" %(it, ELBO)

                if np.abs(ELBO - ELBO_init) < thresh:
                    print 'Convergence'
                    break

                ELBO_init = ELBO

            if it == MAX_ITER:
                break

            it += 1

        #######################################

        self.mu_a = mu_a
        self.cov_a = cov_a
        self.mu_b_e = mu_b_e
        self.cov_b_e = cov_b_e




    def predict(self, K_test):
        N_test = K_test.shape[1]
        P = K_test.shape[0]
        v = self.v

        pred_mu_g = cstack([dot(K_test[:,i,:], self.mu_a) for i in range(N_test)])
        pred_cov_g = np.array([1 + K_test[i,:,:].dot(self.cov_a).dot(K_test[i,:,:].T) for i in range(P)])
        #print pred_mu_g

        #pred_mu_f = mu_b_e[1:].dot(pred_mu_g)+ mu_b_e[0]
        z = np.vstack((np.ones(N_test),pred_mu_g))
        pred_mu_f = z.T.dot(self.mu_b_e)
        #print pred_mu_f

        pred_var_f = 1. + diag(z.T.dot(self.cov_b_e).dot(z))

        z = 1 - norm.cdf((-pred_mu_f + v)/pred_var_f)
        Z_test = z + norm.cdf((-pred_mu_f - v)/pred_var_f)

        #z =  norm.cdf((pred_mu_f - v)/pred_var_f)
        #Z_test = z + norm.cdf((-pred_mu_f - v)/pred_var_f)

        #print z
        prob_y = Z_test**(-1)*(z)
        pred_y = np.zeros_like(prob_y)
        pred_y[prob_y > .5] = 1.
        pred_y[prob_y <= .5] = -1.

        return pred_y, prob_y
