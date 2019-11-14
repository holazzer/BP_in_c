#include<stdio.h>
#include<string.h>
#include<malloc.h>
#include<time.h>
#include<math.h>
#include<stdlib.h>

#define RAND()  ((rand() % 1000) / 1000.0)

typedef struct Node{ double* weights; int length; double threshold;} Node;
typedef struct Layer{ int length; Node** nodes;} Layer;
typedef struct Net{ Layer* hidden; Layer* out;} Net;
typedef struct Temp { double* alpha; double* b; double* beta; } Temp;

typedef enum {STD_BP,ACC_BP,CHG_BP} loss_t;

int std_bp(double** x, double** y,double eta,Net* net,int batch_size,int hidden,int iter);
// int acc_bp(double** x, double** y,double eta,Net* net,int batch_size);
// int chg_bp(double** x, double** y,double eta,Net* net,int batch_size);

double sigmoid(double x){return 1 /( 1 + exp(-x) );}
double dot(double* x,double* y,int size);
double ugly_dot(Net* net,double* g,int h);
double dist_l2(double* x,double* y,int size);
double loss_eval_std(double* x,double* y,int size);

Net* net_new(int hidden_num,int out_num);
int net_random_init(Net* net,int input_num);
int net_save(Net* net,FILE* f);
int net_del(Net* net);
double* net_eval(Net* net,double* x,Temp* temp);
Net* net_load(FILE* f);
int net_update(Net* net,double eta,double* g,double* e,double* x,Temp* t);
double net_check_one(Net* net,double* x,double* y,Temp* t);

int net_parade(Net* net,double**x,double**y,int batch_size,Temp* t);

int layer_save(Layer* layer,FILE* f);

Node* node_new(int x);
int node_del(Node* n);
int node_save(Node* n,FILE* f);


Temp* temp_new(int hidden,int out);
int temp_del(Temp* t);



int main(int argc,char** argv){
    
    // Handling options
    FILE* f = NULL;
    int mode = STD_BP;
    if(argc>=2){
        if(strcmp(argv[1],"stdin")==0){
            printf("Expecting input from stdin.First input x_size,y_size,batch_size,then for each line, x's and y's.\n");
            f = stdin;
        }else{
            f = fopen(argv[1],"r");
            if(f==NULL){
                printf("Fail to open '%s'.\n",argv[1]);
                return -1;
            }
        }
        if(argc==3){
            if(strcmp(argv[2],"acc")==0){printf("Using accumulative bp.\n");mode=ACC_BP;}
            if(strcmp(argv[2],"chg")==0){printf("Using change eta bp.\n");mode=CHG_BP;}
        }
    }



    // get size of vector x, y,batch_size, hidden_layer size, eta
    int x,y,l,q,iter;
    double eta;

    fscanf(f,"%d,%d,%d,%d,%lf,%d",&x,&y,&l,&q,&eta,&iter);

    printf("Got arguments: X:%d,Y:%d,Batch-size:%d,hidden_layer:%d,eta:%f,iter:%d\n",x,y,l,q,eta,iter);

    // Allocating memory for data input.

    double** vxs = malloc(sizeof(double*)*l);
    double** vys = malloc(sizeof(double*)*l);

    int i=0,j=0;
    for(i=0;i<l;i++){
        vxs[i] = malloc(sizeof(double)*x);
        for(j=0; j < x ;j++){
            fscanf(f,"%lf,",vxs[i]+j);
            // printf("Got input: %lf\n",vxs[i][j]);
        }
        vys[i] = malloc(sizeof(double)*y);
        for(j = 0;j<y;j++){
            fscanf(f,"%lf,",vys[i]+j);
            // printf("Got input: %lf\n",vys[i][j]);
        }
    }
    
    fclose(f);

    srand(0);


    Net* net = net_new(q,y);
    net_random_init(net,x);
    std_bp(vxs,vys,eta,net,l,q,iter);
    // save the model
    FILE* n = fopen("NET.net","w");
    net_save(net,n);
    printf("Net saved to 'Net.net'.\n");
    fclose(n);
    net_del(net);

    return 0;

}

// return a pointer of an empty Net.
Net* net_new(int hidden_num,int out_num){
    printf("New net.\n");
    Net* net = malloc(sizeof(Net));
    net->hidden = malloc(sizeof(Layer));
    net->hidden->length = hidden_num;
    net->hidden->nodes=malloc(sizeof(Node*) * hidden_num);
    net->out = malloc(sizeof(Layer));
    net->out->length = out_num;
    net->out->nodes= malloc(sizeof(Node*) * out_num);
    return net;
}


Node* node_new(int x){
    Node* n = malloc(sizeof(Node));
    n->length = x;
    n->weights = malloc(sizeof(double) * x);
    n->threshold = 0;
    return n;
}

int net_random_init(Net* net,int input_num){

    printf("Net init.\n");
    int i,j;
    Node* p = NULL;
    printf("1");

    // hidden layer
    for(i= 0;i<net->hidden->length;i++){
        printf("2");
        p = node_new(input_num);
        printf("3");
        net->hidden->nodes[i] = p;
        printf("4");
        p->threshold = RAND();
        printf("5");
        for(j=0;j<p->length;j++){
            p->weights[j] = RAND();
        }
    }
    printf("Hidden layer.\n");

    // output layer
    for(i = 0;i < net->out->length;i++){
        p = node_new(net->hidden->length);
        net->out->nodes[i] = p;
        p->threshold = RAND();
        for(j = 0;j<p->length;j++){
            p->weights[j] = RAND();
        }
    }

    printf("End of init.\n");
    return 1;

}

// one round of std_bp
int std_bp(double** x, double** y,double eta,Net* net,int batch_size,int hidden,int iter){
    printf("Entering STD_BP.\n");
    int i,j,k;
    Temp* t  = temp_new(net->hidden->length,net->out->length);
    double* g = malloc(sizeof(double) * net->out->length);
    double* e = malloc(sizeof(double) * net->hidden->length);

    for(k=0;k<iter;k++){
        printf("Round %d\n",k);
        for(i=0;i<batch_size;i++){
            double* y_est = net_eval(net,x[i],t);
            // loss
            printf("loss: %f\n",loss_eval_std(y[i],y_est,net->out->length));
            for(j=0;j<net->out->length;j++)g[j] = y_est[j] * ( 1 - y_est[j] ) * ( y[i][j] - y_est[j]);
            for(j=0;j<net->hidden->length;j++) e[j] = t->b[j] * ( 1 - t->b[j]) * ugly_dot(net,g,j);

            net_update(net,eta,g,e,x[i],t);
            // printf("Point %d.\n",i);
        }
    }

    net_parade(net,x,y,batch_size,t);

    
    temp_del(t);

    return 1;
    
}

// Note that y might be a vector, not just a number. 
// So we'are returning a new array of double
// Be careful, the alpha , b, beta also gets calculated here
// Since they are very useful, why not use a struct to store them ? Hence the Temp*.
double* net_eval(  Net* net,  double* x,Temp* t){
    int i;
    // doing alpha and b
    for(i=0;i<net->hidden->length;i++){
        t->alpha[i] = dot(net->hidden->nodes[i]->weights,x,net->hidden->nodes[i]->length);
        t->b[i] = sigmoid( t->alpha[i] -  net->hidden->nodes[i]->threshold );
    }
    // doing beta
    for(i=0;i<net->out->length;i++){
        t->beta[i] = dot( net->out->nodes[i]->weights,t->b,net->out->nodes[i]->length);
    }
    // doing y_estimated
    double* y_est = malloc(sizeof(double) * net->out->length);
    for(i=0;i<net->out->length;i++){
        y_est[i] = sigmoid( t->beta[i] - net->out->nodes[i]->threshold ); 
    }

    return y_est;
}

double dot(  double* x,  double* y,int size){
    double sum = 0;
    for(int i = 0;i<size;i++){
        sum += (x[i] * y[i]);
    }
    return sum;
}

Temp* temp_new(int hidden,int out){
    Temp* t = malloc(sizeof(Temp));
    t->alpha = malloc(sizeof(double) * hidden);
    t->b = malloc(sizeof(double) * hidden);
    t->beta= malloc(sizeof(double) * out);
    return t;
}

// Messy. Because wee need to sum on j , but weights are indexed by h.
double ugly_dot(  Net* net,   double* g,int h){
    double sum = 0;
    for(int i = 0;i<net->out->length;i++){
        net->out->nodes[i]->weights[h] * g[i];
    }
    return sum;
}



int net_update(  Net* net,double eta,double* g,double* e,double* x,Temp* t){
    int i,j,k;
    // v and \gamma
    for(i=0;i<net->hidden->length;i++){
        for(j=0;j<net->hidden->nodes[i]->length;j++){
            net->hidden->nodes[i]->weights[j] += eta * e[i] * x[j];
        }
        net->hidden->nodes[i]->threshold += -eta * e[i];
    }
    // w and \theta
    for(i=0;i<net->out->length;i++){
        for(j=0;j<net->out->nodes[i]->length;j++){
            net->out->nodes[i]->weights[j] += eta * g[i] * t->b[j] ;
        }
        net->out->nodes[i]->threshold += -eta * g[i];
    }
    return 1;
}


double net_check_one(  Net* net,double* x,double* y,Temp* t){
    // Run it on test set to see how well it works.
    double* y_est = net_eval(net,x,t);
    // double E = dist_l2(y,y_est,net->out->length);
    return y_est[0];
}


double dist_l2(  double* x,  double* y,int size){
    double sum = 0;
    for(int i = 0;i<size;i++){
        sum += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return sum;
}

// Only works with output size one.
int net_parade(  Net* net,double**x,double**y,int batch_size,Temp* t){
    int i = 0, right = 0,flag = 0 ;
    double* scores = malloc(sizeof(double) * batch_size);
    for(i=0;i<batch_size;i++){
        scores[i] = net_check_one(net,x[i],y[i],t);
    }
    printf("The test results are:\n");
    for(i=0;i<batch_size;i++){
        flag = ( scores[i] < 0.5 ?  0 : 1 ) == y[i][0];
        right += flag;
        printf("%d:\t%f\t%d\n",i,scores[i],flag);
    }
    printf("accuracy:%d/%d\n",right,batch_size);
    return 1;
}


int net_del(Net* net){
    for(int i = 0;i<net->hidden->length;i++){free(net->hidden->nodes[i]->weights);}
    free(net->hidden->nodes);
    free(net->hidden);
    for(int i = 0;i<net->out->length;i++){free(net->out->nodes[i]->weights);}
    free(net->out->nodes);
    free(net->out);
    free(net);
    return 0;
}

// Nothing. I kinda already have the job done in net_del.
// No need to write one for now, I think.
int node_del(Node* n){
    
}

int temp_del(Temp* t){
    free(t->alpha);
    free(t->beta);
    free(t->b);
    free(t);
    return 0;
}


int net_save(Net* net,FILE* f){
    layer_save(net->hidden,f);
    layer_save(net->out,f);
}


int layer_save(Layer* layer,FILE* f){
    fprintf(f,"Layer:\n");
    fprintf(f,"%d\n",layer->length);
    for(int i = 0;i<layer->length;i++){
        node_save(layer->nodes[i],f);
    }
    return 0;
}


int node_save(Node* n,FILE* f){
    fprintf(f,"Node:\n");
    fprintf(f,"%d,%f\n",n->length,n->threshold);
    for(int i = 0;i<n->length;i++){
        fprintf(f,"%f,",n->weights[i]);
    }
}

double loss_eval_std(double* y,double* y_est,int size){
    double sum = 0;
    for(int i = 0;i<size;i++){
        sum += (y[i] - y_est[i])*(y[i] - y_est[i]);
    }
    return sum / 2;
}