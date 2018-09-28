
console.log('in app.js');


function createTensor() {

    const values = [];
    for (let i = 0; i < 100000; i++) {
        values[i] = Math.random() * 100
    }

    const shape = [200, 500];
    const a = tf.tensor2d(values, shape, 'int32');
    const b = tf.tensor2d(values, shape, 'int32');
    const c = a.add(b);

    //a.print();
    //b.print();
    //c.print();

    c.data().then( (d) => {
        //console.log('yooyouyo', d);
    } );

    // a.dispose();
    // b.dispose();
    // c.dispose();

    //tf.tidy();

}


const model = tf.sequential();

const hidden = tf.layers.dense({
    units: 4,
    inputShape: [2],
    activation: 'sigmoid'
});


const output = tf.layers.dense({
    units: 3,
    activation: 'sigmoid'
});

model.add(hidden);
model.add(output);

model.compile({
    optimizer: tf.train.sgd(0.1),
    loss: tf.losses.meanSquaredError
})


const inputs = tf.tensor2d([
    [0.25, 0.92],
    [0.45, 0.13],
    [0.25, 0.92],
    [0.45, 0.13]
]);

let prediction = model.predict(inputs);
prediction.print();


const outputs = tf.tensor2d([
    [0.1, 0.3, 0.4],
    [0.1, 0.6, 0.4],
    [0.6, 0.2, 0.4],
    [0.5, 0.3, 0.4]
]);

train().then(() => {
    console.log('training done');
    let outputs = model.predict(inputs);
    outputs.print();
    console.log('prediction done');
})

async function train() {
    for (let i = 0; i < 100; i++) {
        const response = await model.fit(inputs, outputs, { shuffle: true });
        console.log(response.history.loss[0]);
    }
}

