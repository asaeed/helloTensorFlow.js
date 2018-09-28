
console.log(data.entries.length);

let model, xs, ys;
var lossP, labelP, rSlider, gSlider, bSlider;

const labelList = [
    'red-ish',
    'green-ish',
    'blue-ish',
    'orange-ish',
    'yellow-ish',
    'pink-ish',
    'purple-ish',
    'brown-ish',
    'grey-ish'
];

function setup() {
    // prepare DOM
    labelP = createP('label: ');
    lossP = createP('loss: ');
    
    rSlider = createSlider(0, 255, 127);
    gSlider = createSlider(0, 255, 127);
    bSlider = createSlider(0, 255, 127);

    // prepare data for input/output layers
    let colors = [];
    let labels = [];
    for (let rec of data.entries) {
        let col = [rec.r / 255, rec.g / 255, rec.b / 255];
        colors.push(col);
        labels.push(labelList.indexOf(rec.label));
    }

    // create input layer tensor
    xs = tf.tensor2d(colors);

    // create output layer tensor
    let labelsTensor = tf.tensor1d(labels, 'int32');
    ys = tf.oneHot(labelsTensor, 9);
    labelsTensor.dispose();

    xs.print();
    ys.print();

    // create the model
    model = tf.sequential();

    // make hidden layer
    let hidden = tf.layers.dense({
        units: 16,
        activation: 'sigmoid',
        inputDim: 3
    })

    let output = tf.layers.dense({
        units: 9,
        activation: 'softmax'
    });

    model.add(hidden);
    model.add(output);

    // create an optimizer
    const optimizer = tf.train.sgd(.2);

    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy'
    });

    // train the model
    train().then(result => {
        console.log(result.history.loss);
    });
}

async function train() {
    var fitOptions = {
        epochs: 20,
        validationSplit: 0.1,
        shuffle: true,
        callbacks: {
            onTrainBegin: () => console.log('training start'),
            onTrainEnd: () => console.log('training complete'),
            onEpochEnd: (num, logs) => {
                console.log('epoch: ' + num);
                lossP.html('loss: ' + logs.loss)
            },
            onBatchEnd: tf.nextFrame // not needed anymnore it seems?
        }
    };

    return await model.fit(xs, ys, fitOptions);
}

function draw() {
    let r = rSlider.value();
    let g = gSlider.value();
    let b = bSlider.value();

    background(r, g, b);

    tf.tidy(() => {
        const xs = tf.tensor2d([
            [r/255, g/255, b/255]
        ]);
        let result = model.predict(xs);
        let index = result.argMax(1).dataSync()[0];

        let label = labelList[index]
        labelP.html('label: ' + label);
    });
    

    // scanline to see if things are still running
    stroke(255);
    strokeWeight(4);
    line(frameCount % width, 0, frameCount % width, height);
}