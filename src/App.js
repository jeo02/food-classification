import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import 'bootstrap/dist/css/bootstrap.css';
import {Card, Tabs, Tab, Button, ListGroup, Table, Badge, Spinner} from 'react-bootstrap'

//In the Future would've liked to put this in a JSON file
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
  //Apps states
  const [isModelLoading, setIsModelLoading] = useState(false)
  const [model, setModel] = useState(null)

  //The queue holds every image the user has added, each with its own status:
  //'queued'     -> waiting to be picked up by the next "Start Processing" click
  //'processing' -> currently being run through the model (part of the active batch)
  //'done'       -> finished, results available
  //'error'      -> classification failed
  const [queue, setQueue] = useState([])

  const fileInputRef = useRef()
  const nextIdRef = useRef(0)

  //Method to load model
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

  //Helper to patch a single queue item by id without touching the rest of the queue.
  const updateQueueItem = (id, changes) => {
      setQueue(prev => prev.map(item => item.id === id ? { ...item, ...changes } : item))
  }

  //Loads an <img> element from an object URL so it can be handed to tf.browser.fromPixels.
  const loadImageElement = (src) => new Promise((resolve, reject) => {
      const img = new Image()
      img.crossOrigin = 'anonymous'
      img.onload = () => resolve(img)
      img.onerror = reject
      img.src = src
  })

  //Adding image(s) to the queue. These always land as 'queued', even if a
  //previous batch is still processing - they will only run on the next Start click.
  const uploadImage = (e) => {
      const { files } = e.target
      if (!files || files.length === 0) return

      const newItems = Array.from(files).map(file => ({
          id: nextIdRef.current++,
          fileName: file.name,
          url: URL.createObjectURL(file),
          status: 'queued',
          results: null,
          error: null,
      }))

      setQueue(prev => [...prev, ...newItems])
      e.target.value = null //allow selecting the same file again later
  }

  //Removes an item from the queue. Only items that aren't actively processing
  //can be removed since a running prediction can't be safely cancelled.
  const removeQueueItem = (id) => {
      setQueue(prev => prev.filter(item => item.id !== id || item.status === 'processing'))
  }

  //Runs the model against a single queued image and stores its results back on the queue.
  const processItem = async (item) => {
      try {
          const imgElement = await loadImageElement(item.url)

          //pre proccessing the image, attempted to do it as similarily to the way it was done in the model.
          const this_results = tf.tidy(() => {
              const scaleFactor = tf.scalar(255);
              const imageTensor = tf.browser.fromPixels(imgElement);
              const resized_image = tf.image.resizeBilinear(imageTensor, [299,299]);
              const imageTensorFinal = resized_image.div(scaleFactor).expandDims(0);
              const prediction = model.predict(imageTensorFinal);

              //Getting the top 5 predictions.
              const topPreds = tf.topk(prediction, 5, true);
              const topPredsVals = topPreds.values.dataSync();
              const topPredsIndices = topPreds.indices.dataSync();

              const results = []
              for (let i = 0; i < topPredsIndices.length; i++) {
                  results.push({ className: MODEL_CLASSES[topPredsIndices[i]], probability: topPredsVals[i] })
              }
              return results
          })

          updateQueueItem(item.id, { status: 'done', results: this_results })
      } catch (error) {
          console.log(error)
          updateQueueItem(item.id, { status: 'error', error: 'Failed to classify this image.' })
      }
  }

  //Called after clicking "Start Processing". Grabs everything currently queued
  //and runs it as one parallel batch. Anything added afterwards stays 'queued'
  //and waits for the next click of this button.
  const startProcessing = () => {
      const toProcess = queue.filter(item => item.status === 'queued')
      if (toProcess.length === 0 || !model) return

      const idsToProcess = new Set(toProcess.map(item => item.id))
      setQueue(prev => prev.map(item => idsToProcess.has(item.id) ? { ...item, status: 'processing' } : item))

      //Kick every prediction off concurrently rather than awaiting them one at a time.
      toProcess.forEach(item => { processItem(item) })
  }

  const triggerUpload = () => {
      fileInputRef.current.click()
  }

  // Load the model
  useEffect(() => {
      loadModel()
  }, [])

  const queuedCount = queue.filter(item => item.status === 'queued').length
  const processingCount = queue.filter(item => item.status === 'processing').length
  const doneCount = queue.filter(item => item.status === 'done').length

  const statusBadge = (status) => {
      switch (status) {
          case 'queued':
              return <Badge bg="secondary">Queued</Badge>
          case 'processing':
              return <Badge bg="primary"><Spinner as="span" animation="border" size="sm" role="status" aria-hidden="true" /> Processing</Badge>
          case 'done':
              return <Badge bg="success">Done</Badge>
          case 'error':
              return <Badge bg="danger">Error</Badge>
          default:
              return null
      }
  }


  if (isModelLoading) {
      return <h2>Model Loading...</h2>
  }


  return (
    <div className='App'>
        <h1 id = "header">Food Identification</h1>
        {/* Tabs for 3 sections, an about me, camera to submit the picture, and a results tab where all the results are displayed. */}
        <Tabs defaultActiveKey="first" fill>
            <Tab eventKey="first" title="Home">
                <div className='center'>
                    <h1><b>Image Classification</b></h1>
                    <p>With the use of <b>tensorflow</b>, I was able to create a model that uses the InceptionV3 pre-trained model to idenitfy 101 different kinds of food.</p>
                    <p>The data set used to train the final model was the <b>food-101</b> dataset.</p>
                    <p>Using the <b>Bootstrap framework</b> I then created a web application to make a user interface to interact with this model.</p>
                    
                    <Card style={{ width: '50rem'}}>
                        <Card.Img variant="top" src="./food_examples.png" />
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
                    <p>The "Upload Image" tab is where you will add pictures of food to a queue. You can keep adding images at any time, then click <b>Start Processing</b> to classify everything currently queued in parallel. Images added while a batch is running stay queued and are only processed the next time you click Start Processing.</p>
                    <p>Finally, the "Results" tab is where we can view the result of the image classification where you can see several results of pictures taken.</p>
                    <br></br>
                </div>
                
            </Tab>
            <Tab eventKey="second" title={<>Upload Image {queuedCount > 0 && <Badge bg="secondary" pill>{queuedCount}</Badge>}</>}>
                <Card style={{ width: '50rem', marginTop: '20px'}}>
                    <Card.Body>
                        <Card.Title>Image Queue</Card.Title>
                        <Card.Text>
                        Add as many images as you like - they'll be added to the queue below. Click <b>Start Processing</b> to run the model on everything currently queued, in parallel. You can keep adding more images while a batch runs; new images stay queued and will only be processed the next time you click Start Processing.
                        </Card.Text>
                        <div className='inputHolder'>
                            <input type='file' accept='image/*' multiple className='uploadInput' onChange={uploadImage} ref={fileInputRef} />
                            <Button variant="outline-primary" className='button' onClick={triggerUpload}>Add Image(s) to Queue</Button>
                            <Button
                                variant="primary"
                                className='button'
                                id="identifyButton"
                                onClick={startProcessing}
                                disabled={queuedCount === 0 || !model}
                            >
                                Start Processing {queuedCount > 0 && `(${queuedCount})`}
                            </Button>
                        </div>
                        {processingCount > 0 &&
                            <p style={{ marginTop: '10px' }}>
                                <Spinner as="span" animation="border" size="sm" role="status" aria-hidden="true" /> Currently processing {processingCount} image{processingCount > 1 ? 's' : ''} in parallel...
                            </p>
                        }
                    </Card.Body>
                </Card>

                <div className='center'>
                    {queue.length === 0 && <p>No images added yet.</p>}
                    {queue.length > 0 &&
                        <ListGroup className='resultsHolder'>
                            {[...queue].reverse().map((item) => (
                                <ListGroup.Item key={item.id} style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                                    <img src={item.url} alt={item.fileName} style={{ width: '60px', height: '60px', objectFit: 'cover' }} />
                                    <div style={{ flexGrow: 1 }}>
                                        <div>{item.fileName}</div>
                                        {item.status === 'error' && <div style={{ color: 'red', fontSize: '0.85em' }}>{item.error}</div>}
                                    </div>
                                    {statusBadge(item.status)}
                                    {item.status !== 'processing' &&
                                        <Button variant="outline-danger" size="sm" onClick={() => removeQueueItem(item.id)}>Remove</Button>
                                    }
                                </ListGroup.Item>
                            ))}
                        </ListGroup>
                    }
                </div>
            </Tab>
            <Tab eventKey="third" title={<>Results {doneCount > 0 && <Badge bg="success" pill>{doneCount}</Badge>}</>}>
                <div className='center'>
                    {doneCount === 0 && <p>No results yet. Add images to the queue and click Start Processing.</p>}
                    {doneCount > 0 &&
                        <ListGroup className='resultsHolder'>
                            {[...queue].reverse().filter(item => item.status === 'done').map((item) => (
                                <Card key={item.id} style={{ width: '50rem', marginTop: '20px'}}>
                                    <Card.Body>
                                        <Card.Img variant = "top" src = {item.url}></Card.Img>
                                        <Table striped bordered hover>
                                                <thead>
                                                    <tr>
                                                        <th>#</th>
                                                        <th>Food Name</th>
                                                        <th>Confidence</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                {item.results.map((curr_result, index) => {
                                                    return (
                                                        <tr key={index}>
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
                            ))}
                        </ListGroup>
                    }
                </div>
            </Tab>
        </Tabs>
    </div>
    
  );
}

export default App;
