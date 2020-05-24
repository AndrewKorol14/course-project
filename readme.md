Work used Node.js only via console
Open console and use next command for creation CNN, train and make prediction:
    node --experimental-modules app.js

For using Tensorboard:
1. Install Tensorboard on computer globally
2. Run previous command
3. Run next command:
    tensorboard --logdir ./logs  