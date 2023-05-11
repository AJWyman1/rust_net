use std::time::Instant;
use ndarray::*;
use rust_net::{NuralNet, ArcNeuralNet};

fn main() {
    //Compare single-threaded and multi-threaded neural nets and find the average time difference
    match load_data("./data/mnist_train.csv".to_owned())
    {
        Ok((train_labels, train_data)) => 
        {
            match load_data("./data/mnist_test.csv".to_owned())
            {
                Ok((test_labels, test_data)) => 
                    {
                        let mut single_total: f32 = 0.0;
                        let mut multi_total: f32 = 0.0;
                        let mut single_accuracy: f32 = 0.0;
                        let mut multi_accuracy: f32 = 0.0;
                        for i in 0..10
                        {   
                            println!("Iteration {}\n-----------------------------",i);
                            println!("starting single thread");
                            let single_thread = Instant::now();
                            let single_thread_acc = single_threaded_nural_net(train_labels.clone(),train_data.clone(),test_labels.clone(),test_data.clone());
                            let single_thread_total_time = single_thread.elapsed().as_secs_f32();
                            println!("starting multithread");
                            let multi_thread = Instant::now();
                            let multi_thread_acc = multi_threaded_nural_net(train_labels.clone(),train_data.clone(),test_labels.clone(),test_data.clone());
                            let multi_thread_total_time = multi_thread.elapsed().as_secs_f32();

                            println!("Single threaded accuracy: {}\n\tTime elapsed: {}\nMulti-threaded accuracy: {}\n\tTime elapsed: {}",single_thread_acc,single_thread_total_time,multi_thread_acc,multi_thread_total_time);
                            single_total += single_thread_total_time;
                            multi_total += multi_thread_total_time;
                            single_accuracy += single_thread_acc;
                            multi_accuracy += multi_thread_acc;
                      
                        }
                        println!("Final Results\n---------------------------");
                       println!("Average Single threaded accuracy: {}\n\tAverage Time elapsed: {}s\nAverage Multi-threaded accuracy: {}\n\tAverage Time elapsed: {}s",single_accuracy / 10.0, single_total / 10.0, multi_accuracy / 10.0,multi_total / 10.0);
                       println!("Threading is {}s faster on average", (single_total - multi_total)/ 10.0);
                    },
                    Err(error) => println!("{}", error)
            }
        },
        Err(error) => println!("{}", error)
    }
    

    // let train_labels = ArcArray2::from_shape_fn((15,1), |(i,_)| i as u32);
    // let train_data = ArcArray1::from_shape_fn(15, |i| vec![i as f32; 16]);
    // let test_labels = ArcArray2::from_shape_fn((5,1),|(i,_)| i as u32);
    // let test_data = ArcArray1::from_shape_fn(5, |i| vec![i as f32; 16]);
     
    //  let mut arc_net = ArcNeuralNet::new(train_labels, train_data, test_labels, test_data, Vec::<usize>::from([5,10]), 6);
    // arc_net.train(20, 300);
    // println!("End of training");
    // let accuracy = arc_net.test();
    // println!("Threaded Accuracy: {}", accuracy);
}

fn single_threaded_nural_net(train_labels: ArcArray2<u32>, train_data: ArcArray1<Vec<f32>>, test_labels: ArcArray2<u32>, test_data: ArcArray1<Vec<f32>>) -> f32
{
    let mut mnist_net = NuralNet::new(784, 2, 18, 10);

    mnist_net.train(train_labels.clone(), train_data.clone(), 784);
    let accuracy = mnist_net.test(test_labels, test_data, 784);
    println!("Accuracy: {}", accuracy);

    accuracy 
}


fn multi_threaded_nural_net(train_labels: ArcArray2<u32>, train_data: ArcArray1<Vec<f32>>, test_labels: ArcArray2<u32>, test_data: ArcArray1<Vec<f32>>) -> f32
{
               
    let train_len = train_data.len();
    let mut arc_mnist_net = ArcNeuralNet::new(train_labels,train_data, test_labels, test_data, vec![18,18], 10);
    arc_mnist_net.train(100, train_len);

    let accuracy = arc_mnist_net.test();
    println!("Threaded Accuracy: {}", accuracy);

    accuracy
   
}

fn load_data(path: String) -> Result<(ArcArray2<u32>, ArcArray1<Vec<f32>>), String> {
    let mut test;
    match csv::Reader::from_path(path) 
    {
        Ok(csv) => test = csv,
        Err(_) => return Err(String::from("Reading error")),
    }
    let mut labels = Vec::<u32>::new();
    let mut dataset = Vec::<f32>::new();
    let mut dataset_vec = Vec::<Vec::<f32>>::new();
    for record in test.records() 
    {
        if let Ok(rec) = record
            {
                let mut img_vec = Vec::<f32>::new();
                    for i in 0..rec.len() 
                    {
                        match rec.get(i).unwrap().trim().parse() 
                        {
                            Ok(datum) => 
                                {
                                    if i == 0 
                                    {
                                        labels.push(datum)
                                    } 
                                    else 
                                    {
                                        img_vec.push(datum as f32 / 255.0);
                                        dataset.push(datum as f32 / 255.0)
                                    }
                                },
                            Err(_) => 
                                {
                                    return Err(String::from("data parse error"));
                                },
                        }
                    }
                    dataset_vec.push(img_vec);
            }
    }

    if let Ok(label_arr)  = ArcArray2::from_shape_vec((labels.len(), 1), labels)
    {
        if let Ok(dataset_arr) = ArcArray1::from_shape_vec(label_arr.len(), dataset_vec)
        {
            return Ok((label_arr, dataset_arr));
        }
        else{return Err(String::from("data fail"));}
    }
    else{
        return Err(String::from("label fail"));}
}