// Slice the extracted transparent sheet without resampling the artwork.
const sharp = require('sharp');
const fs = require('node:fs/promises');
const path = require('node:path');
const boxes = [
  [70,20,225,282], [350,85,225,234], [640,20,211,245],
  [930,10,243,271], [1200,107,200,240], [1450,35,235,270], [1738,90,192,284],
  [75,303,260,235], [420,315,230,230], [710,264,270,230],
  [1045,310,195,268], [1385,325,217,251], [1673,375,243,257],
  [130,538,267,237], [548,520,220,259], [800,495,248,286],
  [1180,530,221,250], [1525,560,230,223],
];
(async () => {
 const out = path.resolve(__dirname, '../../../public/chorus');
 const sizes = [];
 for (const [i, [left, top, width, height]] of boxes.entries()) {
   const cropped = await sharp(path.join(__dirname, 'transparent-sheet.png')).extract({left, top, width, height}).png().toBuffer();
   // Keep the portrait's connected alpha component, excluding neighboring crop fragments.
   const {data: pixels, info: raw} = await sharp(cropped).ensureAlpha().raw().toBuffer({resolveWithObject:true});
   const seen = new Uint8Array(raw.width * raw.height);
   let largest = [];
   for (let start = 0; start < seen.length; start++) {
     if (seen[start] || pixels[start*4+3] < 16) continue;
     const component = [start]; seen[start] = 1;
     for (let k = 0; k < component.length; k++) {
       const pos = component[k], x = pos % raw.width, y = Math.floor(pos/raw.width);
       for (let dy=-1;dy<=1;dy++) for(let dx=-1;dx<=1;dx++) {
         const nx=x+dx,ny=y+dy,n=ny*raw.width+nx;
         if(nx<0||ny<0||nx>=raw.width||ny>=raw.height||seen[n]||pixels[n*4+3]<16) continue;
         seen[n]=1; component.push(n);
       }
     }
     if(component.length>largest.length) largest=component;
   }
   const keep = new Uint8Array(seen.length);
   for(const pos of largest) {
     const x=pos%raw.width,y=Math.floor(pos/raw.width);
     for(let dy=-2;dy<=2;dy++) for(let dx=-2;dx<=2;dx++) {
       if(x+dx>=0 && x+dx<raw.width && y+dy>=0 && y+dy<raw.height) keep[(y+dy)*raw.width+x+dx]=1;
     }
   }
   for(let pos=0;pos<keep.length;pos++) if(!keep[pos]) pixels[pos*4+3]=0;
   const cleaned = await sharp(pixels,{raw:{width:raw.width,height:raw.height,channels:4}}).png().toBuffer();
   const {data, info} = await sharp(cleaned).trim({threshold: 10}).png().toBuffer({resolveWithObject:true});
   const file = `character-${String(i + 1).padStart(2, '0')}.png`;
   await fs.writeFile(path.join(out,file),data);
   sizes.push({file, width:info.width, height:info.height});
 }
 await fs.writeFile(path.join(__dirname,'characters.json'),JSON.stringify(sizes,null,2)+'\n');
 console.log(sizes.map(s=>`${s.width} / ${s.height}`).join(', '));
})();
