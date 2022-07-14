


from utils import client_model_initialization_single_fl

def create_initial_weights(model_initialization,model_id,model_name,n_class,n_neurons):

    model = model_initialization(1,n_class,n_neurons)[0][0]

    filename = "data/"+ str(model_name) +"/" + str(model_id)

    model.save(filename)



create_initial_weights(client_model_initialization_single_fl,model_id=0,model_name="single_fl",n_class=10,n_neurons=32)