import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import 'bootstrap/dist/css/bootstrap.css';
import {Card, Tabs, Tab, Button, ListGroup, Table} from 'react-bootstrap'

const MODEL_CLASSES = ["apple_pie",
"baby_back_ribs",
"baklava",
"beef_carpaccio",
"beef_tartare",
"beet_salad",
"beignets",
"bibimbap",
"bread_pudding",
"breakfast_burrito",
"bruschetta",
"caesar_salad",
"cannoli",
"caprese_salad",
"carrot_cake",
"ceviche",
"cheesecake",
"cheese_plate",
"chicken_curry",
"chicken_quesadilla",
"chicken_wings",
"chocolate_cake",
"chocolate_mousse",
"churros",
"clam_chowder",
"club_sandwich",
"crab_cakes",
"creme_brulee",
"croque_madame",
"cup_cakes",
"deviled_eggs",
"donuts",
"dumplings",
"edamame",
"eggs_benedict",
"escargots",
"falafel",
"filet_mignon",
"fish_and_chips",
"foie_gras",
"french_fries",
"french_onion_soup",
"french_toast",
"fried_calamari",
"fried_rice",
"frozen_yogurt",
"garlic_bread",
"gnocchi",
"greek_salad",
"grilled_cheese_sandwich",
"grilled_salmon",
"guacamole",
"gyoza",
"hamburger",
"hot_and_sour_soup",
"hot_dog",
"huevos_rancheros",
"hummus",
"ice_cream",
"lasagna",
"lobster_bisque",
"lobster_roll_sandwich",
"macaroni_and_cheese",
"macarons",
"miso_soup",
"mussels",
"nachos",
"omelette",
"onion_rings",
"oysters",
"pad_thai",
"paella",
"pancakes",
"panna_cotta",
"peking_duck",
"pho",
"pizza",
"pork_chop",
"poutine",
"prime_rib",
"pulled_pork_sandwich",
"ramen",
"ravioli",
"red_velvet_cake",
"risotto",
"samosa",
"sashimi",
"scallops",
"seaweed_salad",
"shrimp_and_grits",
"spaghetti_bolognese",
"spaghetti_carbonara",
"spring_rolls",
"steak",
"strawberry_shortcake",
"sushi",
"tacos",
"takoyaki",
"tiramisu",
"tuna_tartare",
"waffles"]

function App() {
  const [isModelLoading, setIsModelLoading] = useState(false)
  const [model, setModel] = useState(null)
  const [imageURL, setImageURL] = useState(null);
  const [results, setResults] = useState([])
  
  
  const imageRef = useRef()
  const fileInputRef = useRef()

  const loadModel = async () => {
      setIsModelLoading(true)
      try {
          const model = await tf.loadLayersModel("model/model.json");
          console.log("done loading")
          setModel(model)
          setIsModelLoading(false)
      } catch (error) {
          console.log(error)
          setIsModelLoading(false)
      }
  }

  const uploadImage = (e) => {
      const { files } = e.target
      if (files.length > 0) {
          const url = URL.createObjectURL(files[0])
          setImageURL(url)
      } else {
          setImageURL(null)
      }
  }

  const identify = async () => {
        console.log("predicting:")
        const scaleFactor = tf.scalar(255);
        const imageTensor = tf.browser.fromPixels(imageRef.current);
        const resized_image = tf.image.resizeBilinear(imageTensor, [299,299]);
        const imageTensorFinal = resized_image.div(scaleFactor).expandDims(0);
        const prediction = await model.predict(imageTensorFinal)
        
        console.log(prediction)

        const topPreds = tf.topk(prediction, 5, true);

        const topPredsVals = topPreds.values.dataSync();
        const topPredsIndices = topPreds.indices.dataSync();
        
        let this_results = []
        for(let i = 0; i < topPredsIndices.length; i++){
            this_results.push({className: MODEL_CLASSES[topPredsIndices[i]], probability: topPredsVals[i], image: imageURL})
        }

        setImageURL(null);
        setResults([this_results, ...results]);
  }

  const triggerUpload = () => {
      fileInputRef.current.click()
  }

  useEffect(() => {
      loadModel()
  }, [])


  if (isModelLoading) {
      return <h2>Model Loading...</h2>
  }


  return (
    <div className='App'>
        <h1 id = "header">Food Identification</h1>
        <Tabs defaultActiveKey="first" fill>
            <Tab eventKey="first" title="Home">
                <div className='center'>
                    <h1><b>Image Classification</b></h1>
                    <p>With the use of <b>tensorflow</b>, I was able to create a model that uses the InceptionV3 pre-trained model to idenitfy 101 different kinds of food.</p>
                    <p>The data set used to train the final model was the <b>food-101</b> dataset.</p>
                    <p>Using the <b>Bootstrap framework</b> I then created a web application to make a user interface to interact with this model.</p>
                    
                    <Card style={{ width: '50rem'}}>
                        <Card.Img variant="top" src=".public/food_examples.png" />
                        <Card.Body>
                            <Card.Title>Food Examples</Card.Title>
                            <Card.Text>
                            The food-101 dataset consists of 1000 photos of 101 different kinds of foods. The dataset conists of images of apple pie, baby back ribs, baklava, beef carpaccio, ramen, ravioli, grilled salmon, lasagna, etc...
                            </Card.Text>
                        </Card.Body>
                    </Card>

                    <br></br>
                    <h1><b>Using the App</b></h1>
                    <p>There are three tabs at the top of the page, you are currenlty reading in the "Home" tab where the project is broadly explained.</p>
                    <p>The "Upload Image" tab is where you will upload pictures of food. After you upload you click identify and it will put the results in the reuslts tab. </p>
                    <p>Finally, the "Results" tab is where we can view the result of the image classification where you can see several results of pictures taken.</p>
                    <br></br>
                </div>
                
            </Tab>
            <Tab eventKey="second" title="Upload Image">
                <Card style={{ width: '50rem', marginTop: '20px'}}>
                    {imageURL && <Card.Img variant = "top" src={imageURL} alt="Upload Preview" crossOrigin="anonymous" ref={imageRef} />}
                    <Card.Body>
                        <Card.Title>Image to Upload</Card.Title>
                        <Card.Text>
                        Here you will see a preview of the image you wish to identify.
                        </Card.Text>
                        <div className='inputHolder'>
                            <input type='file' accept='image/*' capture='camera' className='uploadInput' onChange={uploadImage} ref={fileInputRef} />
                            <Button variant="outline-primary" className='button' onClick={triggerUpload}>Upload Image</Button>
                            {imageURL && <Button variant="outline-primary" className='button' id = "identifyButton" onClick={identify}>Identify Image</Button>}
                        </div>
                    </Card.Body>
                </Card>
                
            </Tab>
            <Tab eventKey="third" title="Results">
                <div className='center'>
                    {results.length > 0 && <ListGroup className='resultsHolder'>
                            {results.map((result, index) => {
                                return (
                                    <Card style={{ width: '50rem', marginTop: '20px'}}>
                                        <Card.Body>
                                            <Card.Img variant = "top" src = {result[index].image}></Card.Img>
                                            <Table striped bordered hover>
                                                    <thead>
                                                        <tr key= {index}>
                                                            <th>#</th>
                                                            <th>Food Name</th>
                                                            <th>Confidence</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                    {result.map((curr_result, index) => {
                                                        return (
                                                            <tr>
                                                                <td>{index + 1}</td>
                                                                <td>{curr_result.className.charAt(0).toUpperCase() + curr_result.className.replace("_"," ").substring(1)}</td>
                                                                <td>{Math.round(curr_result.probability * 1000) / 10}%</td>
                                                            </tr>
                                                        )
                                                    })}
                                        </tbody>
                                        </Table>
                                        </Card.Body>
                                    </Card>
                                   
                                )})}
                        </ListGroup>
                    }
                </div>
            </Tab>
        </Tabs>
    </div>
    
  );
}

export default App;
