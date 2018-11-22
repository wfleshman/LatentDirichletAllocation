#include <stdlib.h>

// draws sample given probabilities
double choice(double* probs, int n)
{
    int i;
    double cumulative = probs[0];
    double p = rand()/(double)RAND_MAX;
    for(i=0; i<n; i++)
    {
        if(p <= cumulative)
            return i;

        else if(i+1 == n)
            return n-1;

        else
            cumulative += probs[i+1];
    }
}

// Trains LDA via Collapsed Gibbs Sampling
void fit(int n_iters,
        int *C_wz, 
        int *C_dz, 
        int *words,
        int *docs,
        int *topics,
        int n_tokens,
        int n_words,
        int n_docs,
        int n_topics,
        double alpha,
        double beta)
{
    // iter:iteration, i:token, w:word, z:topic, d:doc
    int iter,i,w,z,d;

    // probabilities of each topic
    double *probs = (double*) calloc(n_topics, sizeof(double));
    double total;

    // easier to work with sum of words per topic counter
    int *C_z = (int*) calloc(n_topics, sizeof(int));
    for(z=0; z<n_topics; z++)
        for(w=0; w<n_words; w++)
            C_z[z] += C_wz[w*n_topics+z];

    for(iter=0; iter<n_iters; iter++)
    {
        // pass through data
        for(i=0; i<n_tokens; i++)
        {
            // grab token info
            w = words[i];
            d = docs[i];
            z = topics[i];

            // adjust counters
            C_wz[w*n_topics+z] -= 1;
            C_dz[d*n_topics+z] -= 1;
            C_z[z] -= 1;

            // calculate unnormalized probabilities
            total = 0;        
            for(z=0; z<n_topics; z++)
            {
                probs[z] = (C_wz[w*n_topics+z] + beta);
                probs[z] *= (C_dz[d*n_topics+z] + alpha);
                probs[z] /= (C_z[z] + n_words*beta);
                total += probs[z];
            }

            // normalize them
            for(z=0; z<n_topics; z++)
                probs[z] /= total;

            // draw latent variable
            z = choice(probs, n_topics);
            topics[i] = z;

            // update counters
            C_wz[w*n_topics+z] += 1;
            C_dz[d*n_topics+z] += 1;
            C_z[z] += 1;
        }
    }
    // clean up
    free(probs);
    free(C_z);
}