from nn import MLP

# Training params
lr = 0.01
steps = 30

# Neural net
mlp = MLP(3, [4, 4, 1])

# Data
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0]


for i in range(steps):
    ypreds = [mlp(x) for x in xs]

    loss = sum((yout - ygt)**2 for yout, ygt in zip(ypreds, ys))
    print("Step ", i, " Loss: ", round(loss.data, 3))
    
    # Flush out gradients
    for p in mlp.parameters():
        p.grade = 0.0

    loss.backward()
    
    for p in mlp.parameters():
        p.data += - lr * p.grad

print("\nFinal predictions:")
for i in range(4):
    print("Predicted", round(ypreds[i].data, 3), "| ", ys[i], "Ground truth")