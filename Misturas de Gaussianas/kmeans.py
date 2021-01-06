import numpy as np

def kmeans(input_x, num_clusters, tol, max_it):
    '''
    Função que aplica a metodologia KMeans para clusterização de dados.
    Dados de entrada:
        input_x: (np.array) dados de entradas a serem clusterizados
        num_clusters: (int) número de grupos de clusterização 
        tol: (float) tolerância maxima do algoritimo de clusterização
        max_it: número máximo de iterações que o algorítimo poderá percorrer
    Saída:
        output_x: (np.array) dados de entrada classificados em grupos
    '''
    x_min = np.amin(input_x)
    x_max = np.amax(input_x)
    cluster_centers = np.random.uniform(low=x_min, high=x_max, size=(num_clusters, 2))
    output_x = np.zeros((input_x.shape[0], input_x.shape[1]+1))
    output_x[:,:-1] = input_x
    categories = range(num_clusters)
    means = np.zeros((num_clusters, 2))
    
    x = True
    num_it = 0
    
    while x:
        old_cluster_centers = np.copy(cluster_centers)

        for i in range(input_x.shape[0]):
            dist = []
            for cluster_center in cluster_centers:
                dist.append(np.linalg.norm(input_x[i]-cluster_center))
            output_x[i,-1] = dist.index(min(dist))
        
        for category in categories:
            in_category = output_x[output_x[:,-1] == category]
            in_category = in_category[:, :-1]
            means[category] = np.mean(in_category, axis=0)
            cluster_centers[category] = means[category]
        
        num_it += 1

        if ((old_cluster_centers - cluster_centers) <= tol).all() or num_it >= max_it: 
            x = False
            
    return(output_x, cluster_centers)