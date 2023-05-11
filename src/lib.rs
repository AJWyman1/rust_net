use std::{sync::{Arc, mpsc::{self, Sender, Receiver}, RwLock},rc::Rc, cell::RefCell, thread::{self}};
use ndarray::*;
use rand::{self, thread_rng, Rng};
pub struct NuralNet {
    layers: Vec<RefCell<Layer>>,
}

struct Layer {
    weights: RefCell<Array2<f32>>,
    biases: RefCell<Array2<f32>>,
    act_fun: ActivationFunction,
}

enum ActivationFunction{
    Sig, // f(x) = x / (1 + abs(x))
    PosSig,
}

impl NuralNet{
    /// Example
    /// ```
    /// use rust_net::NuralNet;
    /// use ndarray::*;
    /// 
    /// let mut nnet = NuralNet::new(5, 2, 3, 4);
    /// //println!("{:?}", nnet.provide_input(Array2::ones((5,5))));
    /// println!("works:\n{}",nnet.calculate_output(arr2(&[[1.],[-2.],[0.],[-4.],[5.]])));
    /// 
    /// 
    /// assert!(false);
    /// ```
    pub fn new(inputs: i32, num_layers: i32, internal_nodes: i32, outputs: i32) -> Self 
    {
        return NuralNet 
        { 
            layers: Layer::new_vec(inputs, internal_nodes, num_layers, outputs),   
        };
    }

    pub fn calculate_output(&mut self, mut input_arr: Array2<f32>) -> Array2<f32>
    {
        for layer in &self.layers
        {
            input_arr = layer.borrow().get_layer_out(&input_arr);
        }
        return input_arr;
    }

    pub fn cost(desired_output: usize, outputs: &Array2<f32>) -> f32
    {
        let mut cost: f32 = 0.0;
        
        for i in 0..outputs.len() 
        {
            let mut x = 0.0;
            if i == desired_output
            {
                x = 1.0;
            }
            cost += ((outputs.get((i, 0)).unwrap()) - x).powf(2.0);
        }
        return cost;
    }

    pub fn train(&mut self, training_labels: ArcArray2<u32>, dataset: ArcArray1<Vec<f32>>, inputs: usize)
    {   
        for i in 0..dataset.len()
        {
            if let (Ok(img_data), Some(target)) = (Array2::from_shape_vec((inputs, 1), dataset.get(i).unwrap().to_vec()), training_labels.get((i,0)))
            {
                let mut outputs = Vec::<Rc<Array2<f32>>>::new();
                outputs.push(Rc::new(img_data));
                for i in 0..self.layers.len() as usize//&self.layers//.enumerate()
                {  
                        let layer_inputs = self.layers.get(i).unwrap().borrow().get_layer_out(outputs.last().unwrap());//layer.borrow().get_layer_out(&img_data);
                        outputs.push(Rc::new(layer_inputs));
                }

                self.backprop(*target as usize, outputs);
              
            }
        }
    }

    pub fn backprop(&mut self, target: usize, layer_outs: Vec<Rc<Array2<f32>>>)
    {
        let targets: &Vec<usize> = &vec![target];
        let mut cost_vec = None;
        for i in (1..layer_outs.len()).rev()
            {
                cost_vec = Some
                            (
                                self.layers.get(i - 1).unwrap().borrow_mut()
                                .backprop(
                                    targets,
                                    &layer_outs.get(i).unwrap(), 
                                    cost_vec, 
                                    &layer_outs.get(i-1).unwrap()
                                )
                            );
            }
    }

    pub fn test(&self, testing_labels: ArcArray2<u32>, dataset: ArcArray1<Vec<f32>>, inputs: usize) -> f32
    {
        let mut correct = 0.0;
        let mut incorrect = 0.0;
        for i in 0..dataset.len()
        {
            if let (Ok(img_data), Some(target)) = (Array2::from_shape_vec((inputs, 1), dataset.get(i).unwrap().to_vec()), testing_labels.get((i,0)))
            {
                let mut outputs = Vec::<Rc<Array2<f32>>>::new();
                outputs.push(Rc::new(img_data));
                for i in 0..self.layers.len() as usize
                {  
                        let layer_inputs = self.layers.get(i).unwrap().borrow().get_layer_out(outputs.last().unwrap());
                        // println!("{:?}", self.layers.get(i).unwrap().borrow().weights);
                        outputs.push(Rc::new(layer_inputs));
                }

                let final_output = outputs.last().unwrap();
                let mut max = 0.0;
                let mut output_from_network = 0;
                for j in 0..final_output.len()
                {
                    if final_output.get((j,0)).unwrap() > &max
                    {
                        max = *final_output.get((j,0)).unwrap();
                        output_from_network = j;
                    }
                }
                if output_from_network == *target as usize
                {
                    correct = correct + 1.0;
                }
                else 
                {
                    incorrect = incorrect + 1.0;
                }
            } 
        }
        return correct / dataset.len() as f32;
    }
}

fn rand_sign(num: f32) -> f32
    {
        if rand::random(){return -num;}
        num
    }

impl Layer
{
    fn new_vec(num_inputs: i32, internal_nodes: i32, internal_layers: i32, outputs: i32) -> Vec<RefCell<Layer>> 
    {
        let mut network = Vec::new();
        network.push(Layer::new(num_inputs, internal_nodes, false)); //input layer
        for _ in 1..internal_layers 
        {
            network.push(Layer::new(internal_nodes, internal_nodes, false)); //hidden nodes 
        }
        network.push(Layer::new(internal_nodes, outputs, true));
        network
    }
    fn new(num_inputs: i32, nodes: i32, output_layer: bool) -> RefCell<Self>
    {
        RefCell::new(
            Layer
            {
                weights: RefCell::new(Array2::from_shape_fn((nodes as usize, num_inputs as usize), |(_,_)| rand_sign(rand::random::<f32>()))),
                biases: RefCell::new(Array2::from_shape_fn((nodes as usize, 1), |(_,_)| rand::random::<f32>()/rand::random::<f32>())),
                act_fun: if output_layer {ActivationFunction::PosSig} else {ActivationFunction::Sig},
            }
        )
    }

    fn get_costs(target: &Vec<usize>, output: &Array2<f32>, final_layer: bool) -> Vec<f32>
    {
        let mut cost_vec: Vec<f32> = vec![];
        for i in 0..output.nrows()
        {
            let mut expected = 0.0;
            if target.contains(&i)
            {
                expected = 1.0;
            }
            let deriv = output.get((i, 0)).unwrap() * (1.0 - output.get((i, 0)).unwrap());

            let cost;

            if final_layer 
            {
                cost = ((output.get((i, 0)).unwrap())).powf(2.0) - expected;
                // println!("expected: {} I: {} cost: {}",expected,i,cost);
            }
            else 
            {
                cost = ((output.get((i, 0)).unwrap()) - expected) * deriv;
            }

            cost_vec.push(cost);
        }
        // println!("{:?}",cost_vec);
        cost_vec
    }

    fn backprop(&mut self, target: &Vec<usize>, output: &Array2<f32>, costs: Option<Vec<f32>>, inputs: &Array2<f32>) -> Vec<f32>
    {        
        let cost_vec:Vec<f32> = 
            match costs
                {
                    None => {
                        // println!("\n target: {:?}", target);
                        // println!("{:?}",output);
                        Layer::get_costs(target, output, true)
                    },
                    Some(cost) => cost,
                };
        let mut prev_layer_costs: Vec<f32> = vec![0.0; self.weights.borrow().ncols()];
        for i in 0..output.nrows()
        {

            let cost = cost_vec.get(i).unwrap();

            let step = cost * 0.02;

            let bias = self.biases.get_mut().get_mut((i,0)).unwrap();
            *bias = *bias - (step * (output.get((i,0)).unwrap() * (1.0 - output.get((i,0)).unwrap())));

            let mut j: usize = 0;

            for weight in self.weights.borrow_mut().row_mut(i)
            {
                prev_layer_costs[j] += (*weight * cost) *  ((output.get((i,0)).unwrap() * (1.0 - output.get((i,0)).unwrap())));
                *weight = *weight - (step  * inputs.get((j,0)).unwrap());
                j = j + 1;
            }
        }
        return prev_layer_costs;
    }

    fn get_layer_out(&self, inputs: &Array2<f32>)  -> Array2<f32>
    {
        let output: Array2<f32> = ((self.weights.borrow().dot(inputs))) + self.biases.borrow().clone();
        return    output.map(|x| 
                match self.act_fun 
                    {
                        ActivationFunction::Sig => 1.0 / (1.0 + 2.7182_f32.powf(-*x)),
                        ActivationFunction::PosSig => 1.0 / (1.0 + 2.7182_f32.powf(- *x))
                    }
                ) ;
    }
}
//////////////////////////////////////////
/// 
/// 
/// Multi-threaded Neural Net
/// 
/// Uses mini-batch gradient descent
/// 
/// 
//////////////////////////////////////////
const E:f32 = 2.7182;

pub struct ArcNeuralNet
{
    input_layer:Arc<RwLock<ArcLayer>>,
    output_layer:Arc<RwLock<ArcLayer>>,

    train_labels:ArcArray2<u32>, 
    train_data:ArcArray1<Vec<f32>>, 
    test_labels:ArcArray2<u32>, 
    test_data:ArcArray1<Vec<f32>>,

    num_inputs:usize,
}

struct ArcLayer
{
    previous_layer:Option<Arc<RwLock<ArcLayer>>>,
    next_layer:Option<Arc<RwLock<ArcLayer>>>,

    weights:RwLock<ArcArray2<f32>>,
    biases:RwLock<ArcArray2<f32>>,

    num_inputs:usize, 
    num_nodes:usize, 
    
}

impl ArcNeuralNet
{
    /// Example
    /// ```
    /// use rust_net::ArcNeuralNet;
    /// use ndarray::*;
    /// 
    /// let train_labels = ArcArray2::from_shape_fn((15,1), |(i,_)| i as u32);
    /// let train_data = ArcArray1::from_shape_fn(15, |i| vec![i as f32; 8]);
    /// let test_labels = ArcArray2::from_shape_fn((5,1),|(i,_)| i as u32);
    /// let test_data = ArcArray1::from_shape_fn(5, |i| vec![i as f32; 8]);
    /// 
    /// let mut arc_net = ArcNeuralNet::new(train_labels, train_data, test_labels, test_data, Vec::<usize>::from([3,4,5]), 7);
    /// 
    /// println!("Output: \n{}", arc_net.get_output(ArcArray2::from_shape_fn((8,1),|(i,_)| i as f32)));
    /// assert!(false);
    /// ```
    pub fn new(train_labels:ArcArray2<u32>, train_data:ArcArray1<Vec<f32>>, test_labels:ArcArray2<u32>, test_data:ArcArray1<Vec<f32>>, internal_layers:Vec<usize>, num_outputs: usize) -> Self
    {
        let num_inputs = train_data.get(0).unwrap().len();
        let (input_layer, output_layer) = ArcLayer::new(num_inputs, num_outputs, &internal_layers);
        ArcNeuralNet 
        { 
            input_layer, 
            output_layer,
            train_labels,
            train_data,
            test_labels,
            test_data, 
            num_inputs,
        }
    }

    /// Takes and input array2 with shape (num_inputs, 1) and returns an output array2 with shape (num_outputs, 1)
    /// /// Example
    /// ```
    /// use rust_net::ArcNeuralNet;
    /// use ndarray::*;
    /// 
    /// let train_labels = ArcArray2::from_shape_fn((15,1), |(i,_)| i as u32);
    /// let train_data = ArcArray1::from_shape_fn(15, |i| vec![i as f32; 8]);
    /// let test_labels = ArcArray2::from_shape_fn((5,1),|(i,_)| i as u32);
    /// let test_data = ArcArray1::from_shape_fn(5, |i| vec![i as f32; 8]);
    /// 
    /// let mut arc_net = ArcNeuralNet::new(train_labels, train_data, test_labels, test_data, Vec::<usize>::from([3,4,5]), 7);
    /// 
    /// println!("Output: \n{}", arc_net.get_output(ArcArray2::from_shape_fn((8,1),|(i,_)| i as f32)));
    /// 
    pub fn get_output(&self, input:ArcArray2<f32>) -> ArcArray2<f32>
    {
        return self.input_layer.read().unwrap().get_output(input);
    }

    /// Begin training the network adjusting the weights and biases after batch_size inputs
    /// Example
    /// ```
    /// use rust_net::ArcNeuralNet;
    /// use ndarray::*;
    /// 
    /// let train_labels = ArcArray2::from_shape_fn((15,1), |(i,_)| i as u32);
    /// let train_data = ArcArray1::from_shape_fn(15, |i| vec![i as f32; 8]);
    /// let test_labels = ArcArray2::from_shape_fn((5,1),|(i,_)| i as u32);
    /// let test_data = ArcArray1::from_shape_fn(5, |i| vec![i as f32; 8]);
    /// 
    /// let mut arc_net = ArcNeuralNet::new(train_labels, train_data, test_labels, test_data, Vec::<usize>::from([3,4,5]), 7);
    /// 
    /// arc_net.train(5, 10);
    /// let accuracy = arc_net.test();
    /// println!("Threaded Accuracy: {}", accuracy);
    ///  
    pub fn train(&mut self, batch_size:usize, mut epoch:usize)
    {
        while epoch > batch_size
        {
            let (tx, rx) = mpsc::channel::<Vec<(ArcArray2<f32>,ArcArray2<f32>)>>();
            let mut pool = Vec::new();
            let output_layer = self.output_layer.clone();

            //start thread for receiver
            let adjustment_handle = thread::spawn(move || {
                    let adjustments = output_layer.read().unwrap().make_adjustments(rx);

                    //update nodes after done receiving messages
                    output_layer.write().unwrap().adjust_weight_and_bias(adjustments);
            });

            let mut rng = thread_rng();
            let num_inputs = &self.num_inputs;

            for _ in 0..batch_size
            {
                //get a random input from training data
                let index: usize = rng.gen_range(0..self.train_data.len());
                let input_data = self.train_data.get(index).unwrap().to_vec();
                let input_label = *self.train_labels.get((index,0)).unwrap() as usize;
                let data = ArcArray2::from_shape_vec((*num_inputs, 1 as usize),input_data).unwrap();

                let tx1 = tx.clone();
                let input_layer = self.input_layer.clone();
                
                //start thread to find layer outputs and send adjusments to the receiver
                let thread = thread::spawn(move || input_layer.read().unwrap().train(data, input_label, tx1, vec![]));
                pool.push(thread);
            }
            epoch -= batch_size;
            
            //wait for all threads to finish
            for handle in pool{
                handle.join().unwrap();
            }

            drop(tx);
            
            //wait for network adjustments to be made
            adjustment_handle.join().unwrap();
        }
    }

    ///Tests the accuracy of the network on unseen data
    pub fn test(&self) -> f32
    {
        let mut correct = 0.0;

        for i in 0..self.test_data.len()
        {
            if let (Ok(data), Some(target)) = (Array2::from_shape_vec((self.num_inputs, 1), self.test_data.get(i).unwrap().to_vec()), self.test_labels.get((i,0)))
            {
                let outputs = self.get_output(data.into());

                let mut max = 0.0;
                let mut output_index = 0;
                for j in 0..outputs.len()
                {
                    if outputs.get((j,0)).unwrap() > &max
                    {
                        max = *outputs.get((j,0)).unwrap();
                        output_index = j;
                    }
                }
                if output_index == *target as usize
                {
                    correct = correct + 1.0;
                }
            }
        }
        return correct / self.test_data.len() as f32;
    }

    
}

impl ArcLayer
{
    /// Returns a tuple of the input and output layers
    fn new(num_inputs:usize, num_outputs:usize, next_layer:&Vec<usize>) -> (Arc<RwLock<Self>>, Arc<RwLock<Self>>)
    {
        let input_layer = Arc::new(RwLock::new(
            ArcLayer
                {
                    previous_layer: None,
                    next_layer: None,

                    weights: RwLock::new(ArcArray2::ones((0,0))),
                    biases: RwLock::new(ArcArray2::zeros((0,0))),

                    num_inputs,
                    num_nodes: num_inputs,
                }
            ));
        let final_layer_nodes = *next_layer.last().unwrap_or(&num_inputs);

        let mut prev_layer = input_layer.clone();

        for layer_node in next_layer
        {
            let num_inputs = prev_layer.read().unwrap().num_nodes;
            prev_layer = Arc::new(RwLock::new(
                ArcLayer
                    {
                        previous_layer: Some(prev_layer),
                        next_layer: None,

                        weights: RwLock::new(ArcArray2::from_shape_fn((*layer_node, num_inputs), |(_,_)| rand_sign(rand::random::<f32>()))),
                        biases: RwLock::new(ArcArray2::from_shape_fn((*layer_node, 1), |(_,_)| rand::random::<f32>()/rand::random::<f32>())),

                        num_inputs,
                        num_nodes: *layer_node,
                    }
                ));
        }

        let output_layer = Arc::new(RwLock::new(
            ArcLayer 
            {   
                previous_layer:Some(prev_layer),
                next_layer: None, 

                weights: RwLock::new(ArcArray2::from_shape_fn((num_outputs, final_layer_nodes), |(_,_)| rand_sign(rand::random::<f32>()))), 
                biases: RwLock::new(ArcArray2::from_shape_fn((num_outputs, 1), |(_,_)| rand::random::<f32>()/rand::random::<f32>())),

                num_inputs: final_layer_nodes, 
                num_nodes: num_outputs, 
            }
        ));
        
        //update layers' next_layer field
        let mut next_layer = output_layer.clone();
        while  next_layer.read().unwrap().previous_layer.is_some()
        {
            let prev_layer =  next_layer.clone().write().unwrap().previous_layer.as_mut().unwrap().clone();
            prev_layer.write().unwrap().next_layer = Some(next_layer.clone());
            next_layer = prev_layer.clone();
        }

        (input_layer, output_layer)
    
    }

    /// Provide an input Array and return the network's final output
    fn get_output(&self, input:ArcArray2<f32>) -> ArcArray2<f32>
    {
        let layer_output:ArcArray2<f32>;
        if self.previous_layer.is_none()
        {
            layer_output = input;
        }
        else 
        {
            layer_output = (((self.weights.read().unwrap().dot(&input)))  + self.biases.read().unwrap().clone()).map(|x| (1.0 / (1.0 + E.powf(-*x)))).into();
        }

        // recursive calls through the layers until output layer is reached
        if let Some(next) = &self.next_layer 
        {
            return next.read().unwrap().get_output(layer_output);
        }
        else
        {
            return layer_output;
        }
    }

    //output for a single layer
    fn get_layer_output(&self, input:ArcArray2<f32>) -> ArcArray2<f32>
    {
        return (((self.weights.read().unwrap().dot(&input)))  + self.biases.read().unwrap().clone()).map(|x| (1.0 / (1.0 + E.powf(-*x)))).into();
    }

    /// Sends a Vec of all layers adjustments to the reciever
    fn train(&self, input:ArcArray2<f32>, label:usize, tx:Sender<Vec<(ArcArray2<f32>,ArcArray2<f32>)>>, mut outputs:Vec<ArcArray2<f32>>)
    {
        let layer_output:ArcArray2<f32>;

        if self.previous_layer.is_none()
        {
            layer_output = input;
        }
        else 
        {
            layer_output = self.get_layer_output(input);
        }
        outputs.push(layer_output.clone()); //building a vector of each layers outputs to be used in backpropegation cost calculations

        // recursive calls through the layers until output layer is reached
        if let Some(next) = &self.next_layer 
        {
            ArcLayer::train(&next.read().unwrap(), layer_output, label, tx, outputs);
        }
        else
        {
            // Output layer
            let costs: Vec<f32> = ArcLayer::get_costs(&vec![label], outputs.last().unwrap()); //output layer costs
            let adjustments = self.build_adjustments(outputs, costs, vec![]); //adjustments for each weight and bias

            tx.send(adjustments).unwrap();
            drop(tx);
        }
        
    }

    /// Calculate the cost of the outputs from the output layer
    fn get_costs(target: &Vec<usize>, output: &ArcArray2<f32>) -> Vec<f32>
    {
        let mut cost_vec: Vec<f32> = vec![];
        for i in 0..output.nrows()
        {
            let mut expected = 0.0;
            if target.contains(&i)
            {
                expected = 1.0;
            }

            let cost;

            cost = ((output.get((i, 0)).unwrap())).powf(2.0) - expected;
            cost_vec.push(cost);
        }
        cost_vec
    }

    /// sums the adjustments received for the batch
    fn make_adjustments(&self, rx:Receiver<Vec<(ArcArray2<f32>,ArcArray2<f32>)>>) -> Vec<(ArcArray2<f32>,ArcArray2<f32>)>
    {
        let mut adjustments:Vec<(ArcArray2<f32>,ArcArray2<f32>)> = vec![]; 
        
        let mut iter = rx.iter();

        while let Some(new_adjustments) = iter.next()//.recv()
        {
            adjustments = ArcLayer::combine(&adjustments, new_adjustments);
        }

        adjustments
    }

    /// each layer creates tuples of weight and bias adjustments and returns a vector of these tuples
    fn build_adjustments(&self, mut all_outputs:Vec<ArcArray2<f32>>, costs:Vec<f32>, mut adjustment_vec:Vec<(ArcArray2<f32>,ArcArray2<f32>)>) -> Vec<(ArcArray2<f32>,ArcArray2<f32>)>
    {
        if self.previous_layer.is_none()
        {
            adjustment_vec.reverse();
            return adjustment_vec;
        }
        else
        {
            let layer_outputs = all_outputs.pop().unwrap();
            let layer_inputs = all_outputs.last().unwrap();
            let (weight_adjust, bias_adjust, costs) = self.get_adjustments(layer_inputs, &layer_outputs, costs);
            
            adjustment_vec.push((weight_adjust, bias_adjust));
            
            return self.previous_layer.as_ref().unwrap().read().unwrap().build_adjustments(all_outputs, costs, adjustment_vec);
        }
    }

    /// Calculates the desired adjustments for the weights and biases of each layer through backpropegation.
    fn get_adjustments(&self, layer_inputs:&ArcArray2<f32>, layer_outputs:&ArcArray2<f32>, costs:Vec<f32>) -> (ArcArray2<f32>,ArcArray2<f32>,Vec<f32>)
    {
        let mut weight_adjust:ArcArray2<f32> = ArcArray2::<f32>::zeros((self.num_nodes, self.num_inputs));
        let mut bias_adjust:ArcArray2<f32> = ArcArray2::<f32>::zeros((self.num_nodes, 1));
        let mut new_costs:Vec<f32> = vec![0.0; self.num_inputs];

        for i in 0..self.num_nodes
        {
            let cost = costs.get(i).unwrap();
            
            //let bias = *self.biases.read().unwrap().get((i,0)).unwrap();

            *bias_adjust.get_mut((i,0)).unwrap() =  - ((cost * 0.02) *  (layer_outputs.get((i,0)).unwrap() * (1.0 - layer_outputs.get((i,0)).unwrap())));

            let mut j:usize = 0;
            for weight in self.weights.read().unwrap().row(i)
            {
                new_costs[j] += (*weight * cost) * ((layer_outputs.get((i,0)).unwrap() * (1.0 - layer_outputs.get((i,0)).unwrap()))); 

                let this_weight_adjust =   - (cost * 0.02 * layer_inputs.get((j,0)).unwrap());
                                
                *weight_adjust.get_mut((i,j)).unwrap() = this_weight_adjust;
                j = j + 1;
            }
        }
        (weight_adjust, bias_adjust, new_costs)
    }
    
    /// helper function to sum weight and bias adjustment arrays
    fn combine(first:&Vec<(ArcArray2<f32>,ArcArray2<f32>)>, mut second:Vec<(ArcArray2<f32>,ArcArray2<f32>)>) -> Vec<(ArcArray2<f32>,ArcArray2<f32>)>
    {
        match first.len().cmp(&second.len())
        {
            std::cmp::Ordering::Less => second,
            std::cmp::Ordering::Greater => first.to_vec(),

            std::cmp::Ordering::Equal => 
            {
                for i in 0..first.len()
                {
                    let (weight_1, bias_1) = first.get(i).unwrap();
                    let ( weight_2, bias_2) = second.get(i).unwrap();
                    *second.get_mut(i).unwrap() = (((weight_1 + weight_2)).into(), ((bias_1 + bias_2)).into());
                };

                second
            },
        }
    }

    /// Changes weight and bias values of each layer
    fn adjust_weight_and_bias(&mut self, mut adjustment_vec:Vec<(ArcArray2<f32>,ArcArray2<f32>)>)
    {
        if self.previous_layer.is_none()
        {
            return;
        }
        else 
        {
            let (weight_adjust, bias_adjust) = adjustment_vec.pop().unwrap();

            let layer_weights = self.weights.get_mut().unwrap();
            let layer_bias = self.biases.get_mut().unwrap();
            for row in 0..layer_weights.nrows()
            {
                *layer_bias.get_mut((row,0)).unwrap() += bias_adjust.get((row,0)).unwrap();
                for col in 0..layer_weights.ncols()
                {
                    *layer_weights.get_mut((row,col)).unwrap() += weight_adjust.get((row,col)).unwrap();
                }
            }
            self.previous_layer.as_mut().unwrap().write().unwrap().adjust_weight_and_bias(adjustment_vec);
        }
    }
}