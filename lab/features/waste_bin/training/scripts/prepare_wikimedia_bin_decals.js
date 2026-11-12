/* Convert the supplied Wikimedia SVGs to white transparent markings.
 * Usage: node prepare_wikimedia_bin_decals.js INPUT_DIR NEW_OUTPUT_DIR
 * Requires the installed sharp library. Source artwork is CC BY-SA 4.0;
 * generated files are modified versions, see attribution.json.
 */
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

async function main() {
  const [source, output] = process.argv.slice(2);
  if (!source || !output) throw new Error('provide input and new output directories');
  fs.mkdirSync(output, { recursive: false });
  const pages = {
    blue: 'https://commons.wikimedia.org/wiki/File:可回收物标志-2019.svg',
    red: 'https://commons.wikimedia.org/wiki/File:有害垃圾标志-2008.svg',
    green: 'https://commons.wikimedia.org/wiki/File:厨余垃圾标志-2008.svg',
    gray: 'https://commons.wikimedia.org/wiki/File:其他垃圾标志-2019.svg',
  };
  for (const name of Object.keys(pages)) {
    const input = path.join(source, name + '.svg');
    const { data, info } = await sharp(input, { density: 288 })
      .resize(768, 768).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
    for (let i = 0; i < data.length; i += 4) {
      const min = Math.min(data[i], data[i + 1], data[i + 2]);
      const max = Math.max(data[i], data[i + 1], data[i + 2]);
      const alpha = max - min < 35 ? Math.max(0, Math.min(255, (min - 160) * 255 / 95)) : 0;
      data[i] = data[i + 1] = data[i + 2] = 255;
      data[i + 3] = Math.round(alpha * data[i + 3] / 255);
    }
    await sharp(data, { raw: { width: info.width, height: info.height, channels: 4 } })
      .png().toFile(path.join(output, name + '.png'));
  }
  fs.writeFileSync(path.join(output, 'attribution.json'), JSON.stringify({
    author: 'Ginbayashi', license: 'CC BY-SA 4.0',
    license_url: 'https://creativecommons.org/licenses/by-sa/4.0/',
    changes: 'Converted to white transparent raster markings; colored background removed.',
    source_pages: pages,
  }, null, 2) + '\n');
}

main().catch(error => { console.error(error); process.exitCode = 1; });
