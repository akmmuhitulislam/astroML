D = discriminator(base_size=30,input_dim=2, output_dim=1,).cuda()
G=generator(base_size=10,output_dim=2).cuda()

batch_sz = 200
z_dim = 1
dataloader=DataLoader(dataset,batch_size=batch_sz,shuffle=True)

G_optimizer = optim.Adam(generator.parameters(G), lr=0.0001)
D_optimizer = optim.Adam(discriminator.parameters(D), lr=0.001)
BCE_loss = nn.BCELoss().cuda()

index,x_ = next(iter(dataloader))
y_real_, y_fake_ = torch.ones(batch_sz, 1), torch.zeros(batch_sz, 1)
y_real_, y_fake_ = y_real_.cuda(), y_fake_.cuda()


Ninner = 1
train_hist = {}
train_hist = {}
train_hist['D_loss_fake'] = []
train_hist['D_loss_true'] = []
train_hist['D_loss_total'] = []
train_hist['G_loss'] = []



for epoch in range(1000):
    for batch_index,x_ in dataloader:
        z_ = 2*torch.rand((batch_sz, 1))-1
        x_, z_ = x_.cuda(), z_.cuda()

        # update D network
        
        for i in range(Ninner):
          D_optimizer.zero_grad()
          D_real = D(x_)
          D_real_loss = BCE_loss(D_real, y_real_)

          G_ = G(z_)
          D_fake = D(G_)
          D_fake_loss = BCE_loss(D_fake, y_fake_) 

          D_loss = D_real_loss + D_fake_loss 
          D_loss.backward()
          D_optimizer.step()

        # update G network
        for i in range(Ninner):
          G_optimizer.zero_grad()
          G_ = G(z_)
          D_fake = D(G_)
          G_loss = BCE_loss(D_fake, y_real_)
        
          G_loss.backward()
          G_optimizer.step()

    train_hist['D_loss_fake'].append(D_fake_loss.item())
    train_hist['D_loss_true'].append(D_real_loss.item())
    train_hist['D_loss_total'].append(D_loss.item())
    train_hist['G_loss'].append(G_loss.item())

    if(np.mod(epoch,50)==0):
        print("Dloss =",D_loss.detach().cpu().numpy(),";Gloss=",G_loss.detach().cpu().numpy())
        z_ = z_.cuda()
        plotDiscriminant_And_Points(D,x_,G_)


plt.figure()
s=plt.plot(train_hist['D_loss_fake'],c='b')
s=plt.plot(train_hist['D_loss_true'],c='m')
s=plt.plot(train_hist['D_loss_total'],c='r')
s=plt.plot(train_hist['G_loss'],c='k')
s = plt.ylim((0,3))
s = plt.grid()
s=plt.legend(('Dloss_fake','D_loss_true','Discriminator loss','Generator loss'))