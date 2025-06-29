[el mismo código anterior hasta `while cap.isOpened():`]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb).resize((384, 384))
        tensor = transforms.ToTensor()(frame_pil).unsqueeze(0)

        with torch.no_grad():
            pred = modelo(tensor)
            mask = torch.argmax(pred.squeeze(), dim=0).byte().cpu().numpy()

        color_mask = np.zeros((384, 384, 3), dtype=np.uint8)
        color_mask[mask == 1] = [255, 255, 255]     # Vesícula: blanco
        color_mask[mask == 2] = [204, 153, 255]     # Cálculos: lila

        frame_np = np.array(frame_pil)
        combined = np.concatenate((frame_np, color_mask), axis=1)
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        out.write(combined_bgr)

        st.image(combined, caption="Original + Segmentación", use_column_width=True)

[continuar con cap.release(), out.release(), y el resto sin cambios]
